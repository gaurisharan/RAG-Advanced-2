import os
import tempfile
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Pinecone Init
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
INDEX_NAME = "ragreader-v2"

# Embeddings (1024-dim, BAAI model)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}  # Switch to 'cuda' if deploying on GPU
)

def process_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
        os.unlink(tmp.name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

def init_qa_chain():
    llm = ChatMistralAI(
        model="mistral-tiny",
        temperature=0.3,
        mistral_api_key=st.secrets["MISTRAL_API_KEY"]
    )

    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# Streamlit UI
st.set_page_config(page_title="RAG Chat", layout="wide")

with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing..."):
            process_documents(uploaded_files)
            st.session_state.qa_chain = init_qa_chain()
            st.success("Documents processed successfully!")

st.title("📚 RAG Chatbot v2")
st.caption(f"Connected to Pinecone index: `{INDEX_NAME}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask something about your documents..."):
    if "qa_chain" not in st.session_state:
        st.error("Upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke(
                    {"question": prompt},
                    return_only_outputs=True
                )
                st.markdown(f"**Answer**: {response['answer']}")
                st.markdown("**Sources:**")
                for doc in response['source_documents'][:3]:
                    source = os.path.basename(doc.metadata['source'])
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(f"- `{source}` (Page {page})")

        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
