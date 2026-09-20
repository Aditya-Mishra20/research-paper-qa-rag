from __future__ import annotations

import httpx
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Paper QA",
    page_icon="📚",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def refresh_documents() -> list[dict]:
    try:
        response = httpx.get(f"{API_URL}/documents", timeout=20.0)
        response.raise_for_status()
        return response.json().get("documents", [])
    except Exception:
        return []


st.title("📚 Research Paper QA")

with st.sidebar:
    st.header("Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Upload and index", type="primary"):
        payload = [
            ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
            for file in uploaded_files
        ]
        with st.spinner("Uploading and embedding..."):
            try:
                response = httpx.post(f"{API_URL}/upload", files=payload, timeout=180.0)
                response.raise_for_status()
                result = response.json()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
            else:
                for item in result.get("saved", []):
                    status = "Skipped duplicate" if item.get("skipped") else "Indexed"
                    st.success(
                        f"{status}: {item.get('filename')} "
                        f"({item.get('chunks', 0)} chunks)"
                    )
                for error in result.get("errors", []):
                    st.error(error)

    with st.expander("Fetch from arXiv"):
        arxiv_id = st.text_input("arXiv ID", placeholder="2401.12345")
        if st.button("Fetch and index"):
            if not arxiv_id.strip():
                st.warning("Enter an arXiv ID.")
            else:
                with st.spinner("Fetching paper..."):
                    try:
                        response = httpx.post(
                            f"{API_URL}/arxiv/fetch",
                            json={"arxiv_id": arxiv_id.strip()},
                            timeout=240.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                    except Exception as exc:
                        st.error(f"arXiv fetch failed: {exc}")
                    else:
                        paper = result.get("paper", {})
                        ingestion = result.get("ingestion", {})
                        st.success(
                            f"Indexed {paper.get('title', arxiv_id)} "
                            f"({ingestion.get('chunks', 0)} chunks)"
                        )

    st.caption("Uses Gemini free-tier APIs for embeddings and answers.")
    documents = refresh_documents()
    if documents:
        st.subheader("Indexed")
        for doc in documents:
            st.write(doc.get("title") or doc.get("doc_id"))
            if doc.get("authors"):
                st.caption(doc["authors"])
    else:
        st.info("No indexed papers yet.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(source.get("citation", "Source")):
                metadata = source.get("metadata", {})
                st.caption(
                    f"Score: {source.get('score', 0):.3f} | "
                    f"Section: {metadata.get('section', 'Unknown')}"
                )
                st.write(source.get("text", ""))

query = st.chat_input("Ask a question about the indexed papers")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json={"query": query, "top_k": 5},
                    timeout=120.0,
                )
                response.raise_for_status()
                result = response.json()
            except Exception as exc:
                answer = f"Query failed: {exc}"
                sources = []
            else:
                answer = result.get("answer", "Not found in uploaded papers.")
                sources = result.get("sources", [])

        st.markdown(answer)
        for source in sources:
            with st.expander(source.get("citation", "Source")):
                metadata = source.get("metadata", {})
                st.caption(
                    f"Score: {source.get('score', 0):.3f} | "
                    f"Section: {metadata.get('section', 'Unknown')}"
                )
                st.write(source.get("text", ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
