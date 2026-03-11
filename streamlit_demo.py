# UI for demo
import streamlit as st
import requests

# testing
API_URL = "http://localhost:8000/api/v1"

st.title("Document QnA")

# sidebar: file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX", type=["pdf", "docx"])

    if uploaded_file:
        if st.button("Ingest"):
            with st.spinner("Indexing..."):
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                )
                result = response.json()
                st.session_state.namespace = result["namespace"]
                st.success(f"Indexed {result['chunks_indexed']} chunks from {result['pages']} pages")

# main: chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# chat input
if prompt := st.chat_input("Ask a question about your document"):
    if "namespace" not in st.session_state:
        st.error("Please upload a document first")
    else:
        # show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "namespace": st.session_state.namespace,
                        "streaming": False
                    }
                )
                result = response.json()
                answer = result["answer"]
                sources = result["sources"]

                st.write(answer)

                # source citations
                with st.expander("Sources"):
                    for s in sources:
                        st.caption(f"{s['filename']} | Page {int(s['page'])} | Score {s['score']:.3f}")
                        st.write(s["text"])

        st.session_state.messages.append({"role": "assistant", "content": answer})