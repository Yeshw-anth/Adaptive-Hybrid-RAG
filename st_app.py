import streamlit as st
from streamlit_chat import message
import requests

st.set_page_config(page_title="Auto-Adaptive RAG", layout="wide")

st.title("Auto-Adaptive RAG Pipeline")

# Place "Clear Chat History" button at the top-left
if st.button("Clear Chat History"):
    st.session_state.history = []

# Sidebar for document uploading
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Select Model", ["llama3", "phi3"])
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files to upload", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                # Prepare files for the batch endpoint
                files_to_upload = [
                    ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                    for uploaded_file in uploaded_files
                ]
                
                try:
                    # Send all files in a single request
                    response = requests.post("http://127.0.0.1:8000/api/upload", files=files_to_upload)
                    response.raise_for_status()
                    results = response.json()
                    
                    # Display results for each file
                    for result in results:
                        if result["status"] == "success":
                            st.success(f"Successfully processed `{result['filename']}` ({result['node_count']} nodes).")
                        else:
                            st.error(f"Failed to process `{result['filename']}`: {result.get('error', 'Unknown error')}")
                            
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to upload batch: {e}")
    

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Main chat area
chat_container = st.container()

with chat_container:
    for i, chat in enumerate(st.session_state.history):
        if chat["role"] == "user":
            message(chat["content"], is_user=True, key=f"user_{i}")
        else:
            message(chat["content"], key=f"assistant_{i}")

# Chat input at the bottom
if query := st.chat_input("Ask a question:"):
    st.session_state.history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # The backend expects a 'session_id', and model selection is handled by the router.
        payload = {"query": query, "session_id": "streamlit_session", "model": selected_model}
        try:
            with st.spinner("Thinking..."):
                response = requests.post("http://127.0.0.1:8000/api/query", json=payload)
                response.raise_for_status()
                data = response.json()
                
                full_response = data.get("answer", "Sorry, I couldn't find an answer.")
                sources = data.get("sources", [])
                confidence_score = data.get("confidence_score")

                message_placeholder.markdown(full_response)

                # Display sources and confidence score in a structured way
                if sources:
                    with st.expander("Details"):
                        st.json(sources)
                        if confidence_score is not None:
                            st.write(f"**Confidence Score:** {confidence_score:.2%}")
                    
        except requests.exceptions.RequestException as e:
            full_response = f"API Error: {e}"
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"An unexpected error occurred: {e}"
            message_placeholder.error(full_response)

    st.session_state.history.append({"role": "assistant", "content": full_response})