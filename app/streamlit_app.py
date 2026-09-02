import streamlit as st

st.set_page_config(
    page_title="Research Paper QA",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper QA System")
st.markdown("Ask questions about your uploaded research papers using RAG.")

with st.sidebar:
    st.header("Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

query = st.text_input("Enter your research question:", placeholder="e.g. What is the core architecture described in the paper?")

if st.button("Submit Question", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching papers and generating response..."):
            # Placeholder for RAG response
            st.subheader("Answer")
            st.write("This is a placeholder response. Integrate your RAG pipeline here.")

            st.subheader("Retrieved Sources")
            st.info("Sample Paper Section 3.2 - Page 4")
