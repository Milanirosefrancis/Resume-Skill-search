
import streamlit as st
import tempfile
import os
import uuid
import pickle
import re
from typing import List, Tuple
from PyPDF2 import PdfReader

# LangChain (NEW API)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# LangChain Core (LCEL)
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------- Utility functions --------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    try:
        # Write bytes into a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        # Now load PDF correctly
        reader = PdfReader(tmp_path)

        text_pages = []
        for p in reader.pages:
            try:
                text_pages.append(p.extract_text() or "")
            except Exception:
                text_pages.append("")

        text = "\n\n".join(text_pages)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return text


def guess_name_from_text(text: str) -> str:
    """A heuristic to guess candidate name from resume text.
    Strategy:
      - Find the first 120 characters, look for lines with likely name (capitalized words)
      - Fallback: search for lines that look like 'Name: X' or 'Candidate: X'
      - Last fallback: filename or 'Unknown'
    This is heuristic-only. For robust performance use an NER model.
    """
    # 1) search for explicit labels
    patterns = [r"^Name[:\-]\s*(.+)$", r"^Candidate[:\-]\s*(.+)$", r"^CV of\s+(.+)$"]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            name = m.group(1).strip()
            # remove trailing contact info
            name = re.split(r"[|,\n\r\\t]|\(|\)", name)[0].strip()
            if 1 <= len(name) <= 60:
                return name

    # 2) look at top lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:8]
    for ln in lines:
        # if line has 1-3 words and most words start with capital letter -> candidate name
        tokens = ln.split()
        if 1 <= len(tokens) <= 4:
            cap_count = sum(1 for t in tokens if t[:1].isupper())
            if cap_count >= max(1, len(tokens) - 1):
                clean = re.sub(r"[^A-Za-z\- .' ]+", "", ln).strip()
                if 1 <= len(clean) <= 60:
                    return clean

    # 3) fallback to first 'word word' with capitalization anywhere
    m = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
    if m:
        return m.group(1)

    return "Unknown"


# -------------------- Indexing pipeline --------------------

@st.cache_data(show_spinner=False)
def build_docs_from_uploads(uploaded_files) -> Tuple[List[Document], dict]:
    """Return LangChain Documents with metadata and a metadata map.
    metadata_map maps internal_id -> {filename, candidate_name, original_bytes}
    """
    docs: List[Document] = []
    metadata_map = {}

    for up in uploaded_files:
        pdf_bytes = up.read()
        text = extract_text_from_pdf_bytes(pdf_bytes)
        candidate_name = guess_name_from_text(text)

        # create a unique id for the resume
        rid = str(uuid.uuid4())
        metadata = {
            "source": up.name,
            "resume_id": rid,
            "candidate_name": candidate_name,
        }
        metadata_map[rid] = {
            "filename": up.name,
            "candidate_name": candidate_name,
            "bytes": pdf_bytes,
            "full_text": text,
        }

        # create a single Document (we will split into chunks later)
        docs.append(Document(page_content=text, metadata=metadata))

    return docs, metadata_map


def create_vectorstore(docs: List[Document], persist_path: str = None):
    """Split docs into chunks, embed and store in FAISS vectorstore."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs_split: List[Document] = splitter.split_documents(docs)

    # initialize embeddings (OpenAI)
    embeddings = OpenAIEmbeddings()

    # create FAISS vectorstore
    vs = FAISS.from_documents(docs_split, embeddings)

    if persist_path:
        with open(persist_path, "wb") as f:
            pickle.dump(vs, f)
    return vs


def load_vectorstore(persist_path: str):
    with open(persist_path, "rb") as f:
        vs = pickle.load(f)
    return vs


# Query and retrieval helpers

def retrieve_best_resumes_by_name(vectorstore: FAISS, metadata_map: dict, name_query: str, k: int = 5) -> List[Tuple[float, dict, str]]:
    """Search the vectorstore for passages matching the name query and return top-k resume metadata with score.
    Returns list of tuples (score, metadata, snippet)
    """
    if vectorstore is None:
        return []

    docs_and_scores = vectorstore.similarity_search_with_score(name_query, k=k)

    # docs_and_scores: List[(Document, score)]
    results = []
    for doc, score in docs_and_scores:
        rid = doc.metadata.get("resume_id")
        snippet = doc.page_content[:800]
        meta = metadata_map.get(rid, {})
        results.append((score, meta, snippet))

    # Group by resume id and keep best score
    grouped = {}
    for score, meta, snippet in results:
        rid = meta.get("resume_id") if meta else None
        key = rid or "unknown"
        if key not in grouped or score < grouped[key][0]:
            grouped[key] = (score, meta, snippet)

    # return sorted by ascending score (FAISS uses smaller = better depending on metric)
    out = sorted(grouped.values(), key=lambda x: x[0])
    return out


def answer_query_with_retrieval(vectorstore, query: str, k: int = 5):
    if vectorstore is None:
        return "No index available. Upload resumes and build the index first."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # LCEL pipeline (new RAG)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)




# -------------------- Streamlit UI --------------------

st.set_page_config(page_title="RAG Resume Search Chatbot", layout="wide")
st.title("📄 RAG Resume Search Chatbot (LangChain + Streamlit)")
st.write("Upload multiple resume PDFs (bulk), build the index, then ask for a candidate or ask questions.")

with st.sidebar:
    st.header("Upload & Index")
    uploaded_files = st.file_uploader("Upload resume PDFs", type=["pdf"], accept_multiple_files=True)
    persist_path = st.text_input("Optional: local path to save vectorstore (e.g. ./resume_index.pkl)", value="")
    build_index_btn = st.button("Build / Rebuild Index")

    st.markdown("---")
    st.header("Index Actions")
    if st.button("Clear cached index"):
        if os.path.exists("./resume_index.pkl"):
            os.remove("./resume_index.pkl")
            st.success("Removed persisted index at ./resume_index.pkl (if existed). Refresh page to clear cache.")
        else:
            st.info("No persisted index found at ./resume_index.pkl")

# Global session state for vectorstore and metadata
if "vs" not in st.session_state:
    st.session_state.vs = None
if "metadata_map" not in st.session_state:
    st.session_state.metadata_map = {}
if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []

# Build index when clicked
if build_index_btn:
    if not uploaded_files or len(uploaded_files) == 0:
        st.sidebar.error("Please upload at least one PDF before building the index.")
    else:
        with st.spinner("Extracting text and building index — this may take a minute depending on number of resumes..."):
            docs, metadata_map = build_docs_from_uploads(uploaded_files)
            vs = create_vectorstore(docs, persist_path=persist_path if persist_path else None)
            st.session_state.vs = vs
            st.session_state.metadata_map = metadata_map
            st.session_state.uploaded_files_names = [f.name for f in uploaded_files]
        st.sidebar.success(f"Indexed {len(uploaded_files)} files into vectorstore (chunks: {len(vs.index_to_docstore_id) if hasattr(vs, 'index_to_docstore_id') else 'unknown'})")

# Try loading persisted index if available
if st.session_state.vs is None and persist_path and os.path.exists(persist_path):
    try:
        st.session_state.vs = load_vectorstore(persist_path)
        st.success("Loaded persisted vectorstore from disk.")
    except Exception as e:
        st.error(f"Couldn't load persisted vectorstore: {e}")

# Main UI: search / chat
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Search by candidate name")
    name_query = st.text_input("Enter candidate name (or part of it)")
    name_k = st.slider("Max candidates to return", min_value=1, max_value=10, value=5)
    if st.button("Find candidate(s)"):
        if not st.session_state.vs:
            st.error("No index available. Upload resumes and build the index first.")
        else:
            results = retrieve_best_resumes_by_name(st.session_state.vs, st.session_state.metadata_map, name_query, k=name_k)
            if not results:
                st.info("No matching resumes found.")
            else:
                st.write(f"Found {len(results)} candidate(s):")
                for score, meta, snippet in results:
                    st.markdown(f"**{meta.get('candidate_name','Unknown')}**  —  *{meta.get('filename','') }*  —  score: `{score:.4f}`")
                    st.caption(snippet)
                    cols = st.columns([0.2, 0.8])
                    if cols[0].button(f"View {meta.get('resume_id')}", key=f"view_{meta.get('resume_id')}"):
                        # show full text and offer download
                        st.write("**Full extracted text:**")
                        st.text_area("Resume text", value=st.session_state.metadata_map[meta['resume_id']]['full_text'], height=400)
                        st.download_button("Download original PDF", data=st.session_state.metadata_map[meta['resume_id']]['bytes'], file_name=meta['filename'])

with col2:
    st.subheader("Ask questions over the indexed resumes")
    user_q = st.text_area("Type your question (e.g. 'Which candidates have Python experience?')")
    q_k = st.slider("Retrieval k (how many passages to search)", min_value=1, max_value=10, value=4)
    if st.button("Ask"):
        if not st.session_state.vs:
            st.error("No index available. Upload resumes and build the index first.")
        else:
            with st.spinner("Running retrieval + LLM..."):
                answer = answer_query_with_retrieval(st.session_state.vs, user_q, k=q_k)
            st.markdown("**Answer:**")
            st.write(answer)

# Show upload summary
st.markdown("---")
st.subheader("Upload / Index summary")
if st.session_state.uploaded_files_names:
    st.write("Uploaded files:")
    for fname in st.session_state.uploaded_files_names:
        st.write(f"- {fname}")

if st.session_state.metadata_map:
    st.write("Indexed candidates (heuristic names):")
    for rid, m in st.session_state.metadata_map.items():
        st.write(f"- {m.get('candidate_name','Unknown')} — {m.get('filename')}  (id: {rid})")

st.caption("Tip: For better name detection, standardize the resume header to put the candidate's name on the first line. Consider swapping the heuristic with an NER model for much higher accuracy.")
