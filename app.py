import streamlit as st
from dotenv import load_dotenv
from complyai.rag import RAGEngine

load_dotenv()

st.set_page_config(page_title="ComplyAI | Document Assistant", page_icon="🛡️", layout="wide")

st.title("🛡️ ComplyAI")
st.caption("AI-powered compliance document assistant · Retrieval-Augmented Generation prototype")
st.info(
    "Upload policy or compliance PDFs, then ask questions grounded in their contents. "
    "This is a prototype: verify important compliance decisions against the original documents."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API key", type="password", help="Used only for answer generation.")
    st.caption("Get an API key from Google AI Studio. Do not commit API keys to GitHub.")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Extract PDF text\n2. Split into chunks\n3. Create embeddings\n4. Retrieve relevant passages\n5. Generate a cited answer")

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

engine = st.session_state.engine

left, right = st.columns([1, 1])
with left:
    st.subheader("1. Add knowledge")
    uploads = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs. Scanned/image-only PDFs may need OCR, which this prototype does not include."
    )
    if st.button("Process documents", type="primary", disabled=not uploads):
        try:
            with st.spinner("Extracting text and building the search index..."):
                count = engine.add_pdfs(uploads)
            st.success(f"Processed {len(uploads)} PDF file(s) and indexed {count} text chunks.")
            st.session_state.indexed_files = sorted(engine.file_names)
            st.session_state.messages = []
        except Exception as exc:
            st.error(f"Could not process documents: {exc}")

with right:
    st.subheader("2. Knowledge base")
    if engine.chunk_count:
        st.metric("Indexed text chunks", engine.chunk_count)
        for name in sorted(engine.file_names):
            st.write(f"📄 {name}")
        if st.button("Clear knowledge base"):
            engine.clear()
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.rerun()
    else:
        st.write("No documents indexed yet.")

st.divider()
st.subheader("3. Ask your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View retrieved sources"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Source {i}: {source['file_name']} · Page {source['page']}**"
                    )
                    st.write(source["text"])
                    st.divider()

question = st.chat_input(
    "Ask about access control, incident response, password policy...",
    disabled=(engine.chunk_count == 0)
)

if question:
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and preparing an answer..."):
                    answer, sources = engine.ask(question, api_key=api_key)
                st.markdown(answer)
                with st.expander("View retrieved sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}: {source['file_name']} · Page {source['page']}**")
                        st.write(source["text"])
                        st.divider()
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                st.caption("Check your API key, network connection and model availability.")
