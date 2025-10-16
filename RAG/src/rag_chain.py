from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatGooglePalm  # Google Palm wrapper
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "../data/chroma")

MODEL_NAME = "gemini-2.5-flash"  # Gemini model

class FreeRAGChain:
    def __init__(self, api_key: str):
        self.embeddings = None
        self.db = None
        self.llm = None
        self.qa_chain = None
        self.api_key = api_key
        self._initialize()

    def _initialize(self):
        """Initialize embeddings, vector store, LLM, and QA chain"""
        print("Initializing RAG system...")

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Vector store
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(f"Chroma database not found at {CHROMA_PATH}. Run ingest.py first")
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # LLM
        self.llm = ChatGooglePalm(
            google_api_key=self.api_key,  # <---- FIXED
            model_name=MODEL_NAME,
            temperature=0
        )

        # QA chain
        self.qa_chain = self._create_qa_chain()
        print("RAG system initialized")


    def _create_qa_chain(self):
        """Create the QA chain with a custom prompt"""
        template = """Use the following context to answer the question.
If you cannot find the answer in the context, say "I don't have enough information to answer that question."

Context: {context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    # def query(self, question: str):
        """Query the RAG system"""
        print(f"Searching for: {question}")
        result =  self.qa_chain.invoke({"query": question})

        answer = result.get("result", "No answer found")
        sources = result.get("source_documents", [])

        # Format sources
        source_info = []
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            source_info.append(f"[{i}] {os.path.basename(source)} (page {page})")

        return {
            "answer": answer,
            "source": "\n".join(source_info) if source_info else None
        }

    def query(self, question: str):
        """Query the RAG system"""
        print(f"Searching for: {question}")
        try:
            # use invoke instead of deprecated __call__
            result = self.qa_chain.invoke({"query": question})
        except Exception as e:
            print("Error querying QA chain:", e)
            return {"answer": "Error querying RAG chain", "source": None}

        # Format sources
        answer = result.get("result", "No answer found")
        sources = result.get("source_documents", [])

        source_info = []
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            source_info.append(f"[{i}] {os.path.basename(source)} (page {page})")

        return {"answer": answer, "source": "\n".join(source_info) if source_info else None}


def create_rag_chain(api_key: str):
    """Factory function to create RAG chain"""
    return FreeRAGChain(api_key=api_key)
