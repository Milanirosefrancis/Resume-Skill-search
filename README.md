# Resume-Skill-search
RAG Resume Search Chatbot
Single-file Streamlit app that:
- Accepts bulk resume PDFs upload
- Extracts text from PDFs
- Heuristically extracts candidate name and stores as metadata
- Splits text into chunks
- Builds a vectorstore (FAISS) with OpenAI embeddings via LangChain
- Provides a chat-like QA interface for asking about particular candidates
- Lets user view matched resume text and download original PDF

Requirements (pip):
langchain
openai
streamlit
PyPDF2
faiss-cpu
tqdm
