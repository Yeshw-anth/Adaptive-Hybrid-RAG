import os
import json
import time
import logging
import psutil
import networkx as nx
from pathlib import Path
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def graph_recovery():
    """
    Recovers the graph from a temporary JSON file if it exists.
    This script is intended to be run BEFORE the main application starts
    to avoid file locking issues with the server process.
    """
    try:
        pass
        # --- Surgically terminate other Python processes ---
        # This was an attempt to break file locks, but it may be too aggressive
        # and unnecessary now that `reload=False` is set for the server.
        # Commenting this out as per user suggestion to test if it's still needed.
        # main_script_name = "main.py"
        # current_pid = os.getpid()
        # logging.warning(f"Current process PID: {current_pid}. Searching for lingering processes running '{main_script_name}'.")
        #
        # for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        #     if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
        #         try:
        #             cmdline = proc.info.get('cmdline')
        #             # Only terminate processes that are zombie instances of our main app
        #             if cmdline and any(main_script_name in arg for arg in cmdline):
        #                 logging.warning(f"Found lingering application process with PID: {proc.info['pid']}. Terminating it.")
        #                 p = psutil.Process(proc.info['pid'])
        #                 p.terminate()
        #         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        #             # Process might have already died, or we don't have permission
        #             pass
        #
        # # Give the OS a moment to process the terminations
        # time.sleep(1)
    except Exception as e:
        logging.error(f"An error occurred during process termination: {e}", exc_info=True)

        # Define paths relative to this script's location
        # Assumes this script is in the root directory
        storage_dir = settings.STORAGE_DIR
        graph_path = settings.GRAPH_PATH
        temp_path = storage_dir / "knowledge_graph.temp.json"

        # --- Recovery Logic ---
        if temp_path.exists():
            logging.warning(f"Found temporary graph file at {temp_path}. Attempting recovery.")

            # 1. Load the graph data from the temporary JSON file.
            try:
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                graph = nx.node_link_graph(data)
                logging.info(f"Successfully loaded graph from temporary file. Contains {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            except Exception as e:
                logging.error(f"FATAL: Could not read or parse {temp_path}: {e}. Manual intervention required.")
                return # Stop execution if we can't even read the temp file

            # 2. Attempt to write the recovered graph to the primary .graphml file.
            try:
                # Write to a new file first
                new_graph_path = graph_path.with_suffix('.graphml.new')
                nx.write_graphml(graph, new_graph_path)
                
                # Atomically replace the old file with the new one.
                # This is more robust than deleting and can break some file locks.
                os.replace(new_graph_path, graph_path)
                logging.info(f"Recovery successful: Graph has been atomically moved to {graph_path}")

                # 3. If replace was successful, remove the temporary JSON file.
                try:
                    os.remove(temp_path)
                    logging.info(f"Removed temporary JSON file: {temp_path}")
                except OSError as e:
                    logging.error(f"Could not remove temporary file {temp_path}: {e}. Please remove it manually.")

            except Exception as e:
                logging.error(f"Recovery failed: Could not save graph to primary file {graph_path}: {e}. The temporary file has been kept for the next attempt.", exc_info=True)

        else:
            logging.info("No temporary graph file found. No recovery needed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the graph recovery process: {e}", exc_info=True)