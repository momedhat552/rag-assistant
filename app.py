import os

import streamlit as st

from rag import INDEX_FILE, answer, build_index

st.title("RAG Assistant")
st.caption("Ask questions about the documents in the docs folder. Answers come only from those documents.")


@st.cache_resource
def ensure_index():
    if not os.path.exists(INDEX_FILE):
        build_index()
    return True


ensure_index()

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching and thinking..."):
        try:
            text, sources = answer(question)
            st.write(text)
            st.caption("Retrieved from: " + ", ".join(sources))
        except Exception as e:
            st.error(f"Something went wrong: {e}")