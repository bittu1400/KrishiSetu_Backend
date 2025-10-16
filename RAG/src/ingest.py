import os
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Absolute path to Chroma DB (relative paths can break in deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "../data/chroma")

MODEL_NAME = "llama2"


class FreeRAGChain:
    def __init__(self):
        self.embeddings = None
        self.db = None
        self.llm = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """Initialize embeddings, vector store, LLM, and QA chain."""
        print("🔧 Initializing RAG system...")

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {e}")

        # Ensure Chroma DB exists
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"❌ Chroma database not found at: {CHROMA_PATH}\n"
                "Run 'ingest.py' first to create it."
            )

        # Load Chroma vectorstore
        try:
            self.db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chroma vectorstore: {e}")

        # Initialize LLM
        try:
            self.llm = OllamaLLM(
                model=MODEL_NAME,
                verbose=False,
                temperature=0.1,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}")

        # Create QA chain
        self.qa_chain = self._create_qa_chain()

        print("RAG system initialized successfully.")

    def _create_qa_chain(self):
        """Create the RetrievalQA chain with custom prompt."""
        template = """You are an expert in Nepali agriculture and crop systems.
Use the following context to answer the user's question accurately.
If the answer is not found in the context, say:
"I don't have enough information to answer that question."

Context:
{context}

Question:
{question}

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

    def query(self, question: str):
        """Query the RAG system with error handling and cleaner output."""
        if not question.strip():
            return {"error": "Question cannot be empty."}

        print(f"Searching for: {question}")

        try:
            result = self.qa_chain.invoke({"query": question})
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")

        # Safely extract results
        answer = result.get("result", "No answer returned.")
        sources = result.get("source_documents", [])

        # Format sources with graceful fallback
        source_info = []
        for i, doc in enumerate(sources, 1):
            meta = getattr(doc, "metadata", {})
            source = os.path.basename(meta.get("source", "unknown"))
            page = meta.get("page", "N/A")
            source_info.append(f"[{i}] {source} (page {page})")

        return {
            "answer": answer.strip(),
            "sources": source_info or ["No sources found."]
        }


def create_rag_chain():
    """Factory function to create a FreeRAGChain instance."""
    return FreeRAGChain()


if __name__ == "__main__":
    rag = create_rag_chain()
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        res = rag.query(q)
        print("\nAnswer:", res["answer"])
        print("Sources:")
        for s in res["sources"]:
            print("  -", s)
