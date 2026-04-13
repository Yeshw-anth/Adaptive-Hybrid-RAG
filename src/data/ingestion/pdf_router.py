
import logging
from typing import List, Dict, Any, Optional

from unstructured.documents.elements import Element, ElementMetadata

from src.data.ingestion.base_loader import Loader

logger = logging.getLogger(__name__)

class IntelligentPDFRouter(Loader):
    """
    Intelligently routes PDF documents to the best parsing tool based on analysis.
    Orchestrates a multi-step process:
    1. Analyze: Perform a cheap, initial analysis of the PDF.
    2. Decide: Choose the best tool (PyMuPDF, Camelot, OCR) based on the analysis.
    3. Execute: Run the chosen tool to extract content.
    4. Normalize: Convert the tool's output into a standardized list of `unstructured.Element` objects.
    5. Validate & Fallback: If the primary tool fails, fall back to a safer alternative.
    """

    def __init__(self, file_path: str):
        """
        Initializes the router with the path to the PDF file.
        :param file_path: The absolute path to the PDF file.
        """
        self.file_path = file_path

    def process(self) -> List[Element]:
        """
        Orchestrates the PDF processing workflow using a tiered fallback system.
        1. Tier 1 (Primary): Advanced PyMuPDF extraction with layout analysis.
        2. Tier 2 (Fallback): Simple PyMuPDF text extraction if Tier 1 fails.
        3. Tier 3 (Final Fallback): OCR for scanned documents or complete failures.
        """
        analysis_results = self._analyze_pdf()

        # If analysis determines the document is scanned, go directly to OCR.
        if analysis_results["is_scanned"]:
            logger.info("Document is scanned. Routing directly to OCR.")
            return self._execute_ocr()

        # --- Tier 1: Attempt the primary, high-fidelity extraction pipeline ---
        try:
            logger.info("--- Tier 1: Attempting Advanced PyMuPDF Extraction ---")
            raw_elements = self._execute_pymupdf()
            normalized_elements = self._normalize_pymupdf_output(raw_elements)

            # If tables were detected, extract them and merge with the other elements.
            if analysis_results["pages_with_tables"]:
                logger.info("Extracting tables with Camelot...")
                table_elements = self._extract_tables_with_camelot(pages=analysis_results["pages_with_tables"])
                normalized_elements.extend(table_elements)
                # Re-sort elements by page and vertical position to ensure correct final order
                normalized_elements.sort(key=lambda el: (
                    el.metadata.get("page_num", 0),
                    el.metadata.get("bbox", (0, 0, 0, 0))[1]
                ))
            
            logger.info("--- Tier 1: Advanced PyMuPDF Extraction Successful ---")
            return normalized_elements
        except Exception as e:
            logger.warning(f"--- Tier 1 Failed: {e}. Proceeding to Tier 2. ---", exc_info=True)

        # --- Tier 2: Attempt simple PyMuPDF text extraction as a fallback ---
        try:
            logger.info("--- Tier 2: Attempting Simple PyMuPDF Extraction ---")
            elements = self._execute_pymupdf_simple()
            if elements:
                logger.info("--- Tier 2: Simple PyMuPDF Extraction Successful ---")
                return elements
            else:
                logger.warning("--- Tier 2: Simple extraction yielded no elements. Proceeding to Tier 3. ---")
                raise RuntimeError("Simple PyMuPDF extraction failed to produce elements.")
        except Exception as e:
            logger.error(f"--- Tier 2 Failed: {e}. Proceeding to Tier 3 (OCR). ---", exc_info=True)

        # --- Tier 3: Use OCR as the final resort ---
        logger.info("--- Tier 3: Attempting OCR Extraction ---")
        return self._execute_ocr()

    def _execute_pymupdf_simple(self) -> List[Element]:
        """
        A simple and robust fallback method that extracts plain text from a PDF
        using PyMuPDF, wrapping it in a single NarrativeText element.
        """
        import fitz
        from unstructured.documents.elements import NarrativeText
        
        logger.info("Executing simple PyMuPDF text extraction.")
        doc = fitz.open(self.file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n\n"
        doc.close()

        if full_text.strip():
            metadata = ElementMetadata(filename=self.file_path)
            metadata.extraction_method = "pymupdf_simple"
            return [NarrativeText(text=full_text.strip(), metadata=metadata)]
        return []


    def _analyze_pdf(self) -> Dict[str, Any]:
        """
        Performs a quick analysis of the PDF to guide the extraction strategy.
        Detects if the PDF is scanned (image-based) and identifies pages
        that likely contain tables based on vector graphics.
        """
        import fitz  # PyMuPDF
        logger.info("Analyzing PDF for structure and content type...")
        
        doc = fitz.open(self.file_path)
        analysis_results = {
            "is_scanned": False,
            "pages_with_tables": []
        }
        
        has_text = False
        for page_num, page in enumerate(doc):
            # --- Check for text ---
            if page.get_text("text"):
                has_text = True

            # --- Check for tables via vector graphics ---
            # A simple heuristic: many horizontal/vertical lines suggest a table.
            paths = page.get_drawings()
            h_lines = [p for p in paths if p['rect'].height < 1 and p['rect'].width > 20]
            v_lines = [p for p in paths if p['rect'].width < 1 and p['rect'].height > 20]

            if len(h_lines) > 5 and len(v_lines) > 2:
                # Page numbers are 1-based for Camelot, but 0-based in PyMuPDF
                analysis_results["pages_with_tables"].append(page_num + 1)
        
        # If no text was found anywhere in the document, it's likely scanned
        if not has_text:
            analysis_results["is_scanned"] = True
            logger.info("Analysis complete: PDF is likely scanned (image-based).")
        else:
            logger.info(f"Analysis complete: Found potential tables on pages: {analysis_results['pages_with_tables']}")
            
        return analysis_results

    def _execute_pymupdf(self) -> List[Dict[str, Any]]:
        """
        Extracts content using PyMuPDF, returning a list of raw dictionaries
        for text spans and images, with reading order corrected for multi-column layouts
        using DBSCAN clustering.
        """
        import fitz  # PyMuPDF
        import numpy as np
        from sklearn.cluster import DBSCAN

        logger.info("Extracting content with PyMuPDF (including robust multi-column detection)...")
        doc = fitz.open(self.file_path)
        final_ordered_elements = []

        for page_num, page in enumerate(doc):
            # --- 1. Extract raw text and image elements for the page ---
            page_elements = []
            page_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
            for block in page_blocks:
                if block['type'] == 0:
                    for line in block['lines']:
                        for span in line['spans']:
                            page_elements.append({"type": "text", "page_num": page_num, **span})
            
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bbox = page.get_image_bbox(img)
                page_elements.append({
                    "type": "image", "page_num": page_num, "bbox": img_bbox,
                    "image_bytes": base_image["image"], "image_ext": base_image["ext"]
                })

            if not page_elements or len(page_elements) < 2:
                final_ordered_elements.extend(page_elements)
                continue

            # --- 2. Detect columns using DBSCAN and correct reading order ---
            x_coords = np.array([el['bbox'][0] for el in page_elements]).reshape(-1, 1)
            
            # Use a fraction of the page width as the epsilon for clustering
            # This allows elements close horizontally to be grouped into a column
            epsilon = page.rect.width * 0.05 
            db = DBSCAN(eps=epsilon, min_samples=1).fit(x_coords)
            labels = db.labels_
            
            num_columns = len(set(labels))
            if num_columns > 1:
                logger.info(f"{num_columns}-column layout detected on page {page_num + 1}. Correcting reading order.")
                
                # Group elements by their column (cluster label)
                columns = {label: [] for label in set(labels)}
                for i, el in enumerate(page_elements):
                    columns[labels[i]].append(el)
                
                # Determine the left-to-right order of the columns
                column_order = sorted(
                    columns.keys(),
                    key=lambda label: np.mean([el['bbox'][0] for el in columns[label]])
                )
                
                # Sort elements within each column vertically, then concatenate columns
                ordered_page_elements = []
                for label in column_order:
                    sorted_column = sorted(columns[label], key=lambda el: el['bbox'][1])
                    ordered_page_elements.extend(sorted_column)
                page_elements = ordered_page_elements
            else:
                # Single column: just sort by vertical position
                page_elements.sort(key=lambda el: el['bbox'][1])

            final_ordered_elements.extend(page_elements)
        
        logger.info(f"PyMuPDF extracted {len(final_ordered_elements)} raw text and image elements with corrected order.")
        return final_ordered_elements

    def _normalize_pymupdf_output(self, raw_elements: List[Dict[str, Any]]) -> List[Element]:
        """
        Converts the ordered list of raw elements into a hierarchically grouped
        list of `unstructured.Element` objects (Headers, NarrativeText, Code, Images).
        """
        from unstructured.documents.elements import NarrativeText, Image, Header, ElementMetadata
        try:
            from unstructured.documents.elements import Code
        except ImportError:
            logger.warning("'Code' element not found in `unstructured.documents.elements`. Falling back to `NarrativeText` for code blocks.")
            Code = NarrativeText
        from collections import defaultdict
        import statistics

        if not raw_elements:
            return []

        # Handle case where there are only images
        text_elements = [el for el in raw_elements if el["type"] == "text"]
        if not text_elements:
            images = []
            for el in raw_elements:
                if el["type"] == "image":
                    raw_meta = el.copy()
                    page_number = raw_meta.pop("page_num", None)
                    
                    img_meta = ElementMetadata(filename=self.file_path, page_number=page_number)
                    for key, value in raw_meta.items():
                        setattr(img_meta, key, value)

                    images.append(Image(text="", metadata=img_meta))
            return images

        # --- Step 1: Hierarchical Style Analysis ---
        font_sizes = [el["size"] for el in text_elements]
        baseline_size = statistics.mode(font_sizes) if font_sizes else 12
        
        potential_headers = [el for el in text_elements if el["size"] > baseline_size * 1.05 or "bold" in el["font"].lower()]
        header_styles = defaultdict(int)
        for h in potential_headers:
            style_key = (round(h["size"], 1), h["font"])
            header_styles[style_key] += 1
        
        sorted_styles = sorted(header_styles.keys(), key=lambda s: (-s[0], -header_styles[s]))
        style_to_level = {style: i + 1 for i, style in enumerate(sorted_styles)}

        # --- Step 2: Hierarchical Grouping ---
        final_elements = []
        current_group = None

        def flush_current_group():
            nonlocal current_group
            if not current_group: return
            
            text = current_group["text"].strip()
            if text:
                raw_meta = current_group["metadata"]
                page_number = raw_meta.pop("page_num", None)
                
                metadata = ElementMetadata(filename=self.file_path, page_number=page_number)
                for key, value in raw_meta.items():
                    setattr(metadata, key, value)

                element_props = {"text": text, "metadata": metadata}
                
                if current_group["type"] == "Code":
                    final_elements.append(Code(**element_props))
                else:
                    final_elements.append(NarrativeText(**element_props))
            current_group = None

        for el in raw_elements:
            if el["type"] == "image":
                flush_current_group()
                raw_meta = el.copy()
                page_number = raw_meta.pop("page_num", None)
                
                img_metadata = ElementMetadata(filename=self.file_path, page_number=page_number)
                for key, value in raw_meta.items():
                    setattr(img_metadata, key, value)

                final_elements.append(Image(text="", metadata=img_metadata))
                continue

            # --- Text Processing ---
            text = el["text"].strip()
            if not text: continue

            style_key = (round(el["size"], 1), el["font"])
            element_type = "NarrativeText"
            is_header = False

            if style_key in style_to_level:
                element_type = "Header"
                is_header = True
            elif "mono" in el["font"].lower() or "courier" in el["font"].lower():
                element_type = "Code"

            if is_header:
                flush_current_group()
                raw_meta = el.copy()
                page_number = raw_meta.pop("page_num", None)
                
                header_metadata = ElementMetadata(filename=self.file_path, page_number=page_number)
                header_metadata.level = style_to_level[style_key]
                for key, value in raw_meta.items():
                    setattr(header_metadata, key, value)

                final_elements.append(Header(text=text, metadata=header_metadata))
                continue

            if not current_group or current_group["type"] != element_type:
                flush_current_group()
                current_group = {"type": element_type, "text": text, "metadata": el}
            else:
                current_group["text"] += " " + text
        
        flush_current_group()

        logger.info(f"Normalized and grouped into {len(final_elements)} structured hierarchical elements.")
        return final_elements

    def _execute_ocr(self) -> List[Element]:
        """
        Executes a pure OCR pipeline using PyMuPDF (fitz) and pytesseract
        as a fallback for scanned or un-extractable documents.
        This avoids the external dependency on Poppler required by pdf2image.
        """
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io
        from unstructured.documents.elements import NarrativeText

        logger.warning(f"Falling back to pure OCR for file: {self.file_path} using PyMuPDF and Tesseract.")
        try:
            doc = fitz.open(self.file_path)
            full_text = ""
            for i, page in enumerate(doc):
                try:
                    # Render page to a pixmap (an image)
                    pix = page.get_pixmap(dpi=300)  # Higher DPI for better OCR accuracy
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR on the image
                    text = pytesseract.image_to_string(image)
                    full_text += text + "\n\n"
                except Exception as ocr_err:
                    logger.error(f"OCR failed on page {i + 1} of {self.file_path}: {ocr_err}")
            
            doc.close()

            if full_text.strip():
                # For pure OCR, we can't infer structure, so we return a single NarrativeText element.
                metadata = ElementMetadata(filename=self.file_path)
                metadata.ocr_fallback = True
                return [NarrativeText(text=full_text.strip(), metadata=metadata)]
            else:
                logger.error(f"OCR processing yielded no text for {self.file_path}.")
                return []
        except Exception as e:
            logger.error(f"Error during pure OCR fallback for {self.file_path}: {e}", exc_info=True)
            return []

    def _extract_tables_with_camelot(self, pages: List[int]) -> List[Element]:
        """
        Extracts tables from specified pages using Camelot and normalizes them
        into unstructured.Table elements.
        """
        import camelot
        from unstructured.documents.elements import Table

        logger.info(f"Attempting to extract tables from pages: {pages} with Camelot...")
        table_elements = []
        
        # Camelot requires a string of page numbers, e.g., "1,3,5"
        page_str = ",".join(map(str, pages))

        try:
            # Use 'stream' for tables without clear grid lines, 'lattice' is for bordered tables
            tables = camelot.read_pdf(self.file_path, pages=page_str, flavor='stream')
            
            for table in tables:
                if table.parsing_report['accuracy'] < 90:
                    logger.warning(f"Skipping table on page {table.page} due to low accuracy: {table.parsing_report['accuracy']}%" )
                    continue

                # Convert the pandas DataFrame to an HTML string to preserve structure
                table_html = table.df.to_html(index=False, header=True)
                
                # Get the bounding box of the table on the page
                # Camelot's coordinate system might differ, this is an approximation
                # For more accuracy, we would need to map coordinate systems.
                # For now, we store the raw camelot coordinates.
                raw_coords = table._bbox

                metadata = ElementMetadata(filename=self.file_path, page_number=table.page)
                metadata.camelot_parsing_report = table.parsing_report
                metadata.camelot_bbox = raw_coords

                
                table_elements.append(Table(text=table_html, metadata=metadata))
                logger.info(f"Successfully extracted and normalized a table from page {table.page}.")

        except Exception as e:
            logger.error(f"Error during Camelot table extraction: {e}", exc_info=True)

        return table_elements