import os
import re
import time
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

# Optional torch import for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from groq import Groq


# =========================
# Configuration
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

DATA_DIR = Path("RAG/data")
VECTOR_DIR = Path("vector_store")
VECTOR_STORE_PATH = VECTOR_DIR / "faiss_index"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
TOP_K = 5
MAX_CONTEXT_CHARS = 12000
SIMILARITY_DISTANCE_MAX = 0.6
MMR_FETCH_K = 25

PREFERRED_GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

ALLOWED_LANGS = {"English", "नेपाली", "हिन्दी"}

# Global state
DEVICE = None
GROQ_MODEL = None
vector_store = None

# =========================
# Device Detection
# =========================
def choose_device() -> str:
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("🚀 GPU detected. Using CUDA for embeddings.")
            return "cuda"
    except Exception as e:
        print(f"⚠️ GPU check failed: {e}")
    print("💻 Using CPU for embeddings.")
    return "cpu"


# =========================
# Groq Model Selection
# =========================
def pick_groq_model() -> str:
    try:
        available_models = {m.id for m in groq_client.models.list().data}
        for model in PREFERRED_GROQ_MODELS:
            if model in available_models:
                print(f"✅ Using Groq model: {model}")
                return model
        if available_models:
            fallback = sorted(available_models)[0]
            print(f"✅ Using available model: {fallback}")
            return fallback
    except Exception as e:
        print(f"⚠️ Could not list Groq models: {e}")
    print("✅ Using default model: llama-3.1-8b-instant")
    return "llama-3.1-8b-instant"


# =========================
# Text Processing Utilities
# =========================
def is_junk_content(text: str) -> bool:
    """Filter out tables, catalogs, and low-quality text."""
    if not text or len(text.strip()) < 120:
        return True
    
    t = text.strip()
    letters = sum(1 for c in t if c.isalpha())
    if letters / max(1, len(t)) < 0.40:
        return True
    
    if re.search(r"(,|;)\s*\w+(?:\s*\d+)?(?:\s*,\s*\w+){5,}", t):
        return True
    
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and sum(1 for ln in lines if re.match(r"^[\d\W]+$", ln)) / len(lines) > 0.5:
        return True
    
    return False


def e5_passage(text: str) -> str:
    return f"passage: {text}"


def e5_query(text: str) -> str:
    return f"query: {text}"


def truncate_context(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    mid = limit // 2
    return text[:mid] + "\n...\n" + text[-mid:]


# =========================
# PDF Loading
# =========================
def load_pdfs_from_directory(directory: Path) -> List[Document]:
    """Load all PDFs from a directory with fallback mechanisms."""
    documents = []
    
    if not directory.is_dir():
        print(f"⚠️ Directory not found: {directory}")
        return documents
    
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    if not pdf_files:
        print(f"⚠️ No PDFs found in {directory}")
        return documents
    
    for file_path in pdf_files:
        print(f"📄 Loading {file_path.name}...")
        success = False
        
        # Try PyPDF2 first
        try:
            reader = PdfReader(str(file_path))
            for page_num, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path.name, "page": page_num + 1}
                    ))
                    success = True
        except Exception as e:
            print(f"⚠️ PyPDF2 error on {file_path.name}: {e}")
        
        # Fallback to pdfminer if needed
        if not success:
            try:
                text = (pdfminer_extract_text(str(file_path)) or "").strip()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path.name, "page": "full"}
                    ))
                    success = True
            except Exception as e:
                print(f"⚠️ pdfminer error on {file_path.name}: {e}")
        
        if not success:
            print(f"⚠️ Could not extract text from {file_path.name}")
    
    return documents


# =========================
# Vector Store Management
# =========================
def initialize_vectorstore(force_rebuild: bool = False) -> FAISS:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    print("🔧 Initializing embeddings...")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Try loading existing FAISS index
    if VECTOR_STORE_PATH.exists() and not force_rebuild:
        try:
            print("📂 Loading existing FAISS index...")
            vstore = FAISS.load_local(
                str(VECTOR_STORE_PATH),
                embedder,
                allow_dangerous_deserialization=True
            )

            # ✅ Validate embedding dimension match
            test_vec = embedder.embed_query("dimension check")
            if vstore.index.d != len(test_vec):
                print(f"⚠️ Dimension mismatch (expected {vstore.index.d}, got {len(test_vec)}). Rebuilding...")
                raise ValueError("FAISS dimension mismatch")

            print("✅ Loaded FAISS index successfully.")
            return vstore

        except Exception as e:
            print(f"⚠️ Could not load existing index: {e}. Rebuilding...")

    # 🔁 Rebuild vector store
    print("📄 Loading and embedding documents...")
    docs = load_pdfs_from_directory(DATA_DIR)  # make sure this returns your documents
    print(f"✅ Loaded {len(docs)} documents.")

    vstore = FAISS.from_documents(docs, embedder)
    vstore.save_local(str(VECTOR_STORE_PATH))
    print("✅ Rebuilt and saved new FAISS index.")
    return vstore

# =========================
# Retrieval
# =========================
def retrieve_context(query: str, k: int = TOP_K) -> Tuple[str, List[str]]:
    """Retrieve relevant context with quality filtering."""
    if not query or not query.strip():
        return "", []
    
    q = e5_query(query.strip())
    
    # Retrieve with MMR for diversity
    try:
        candidates = vector_store.max_marginal_relevance_search(
            q, k=k, fetch_k=MMR_FETCH_K, lambda_mult=0.5
        )
    except Exception:
        candidates = vector_store.similarity_search(q, k=k)
    
    # Score filtering
    keep_ids = set()
    try:
        scored = vector_store.similarity_search_with_score(q, k=MMR_FETCH_K)
        keep_ids = {id(doc) for doc, score in scored 
                   if score is not None and score <= SIMILARITY_DISTANCE_MAX}
    except Exception:
        pass
    
    context_chunks = []
    sources = []
    seen = set()
    
    for result in candidates:
        if keep_ids and id(result) not in keep_ids:
            continue
        
        text = (result.page_content or "").strip()
        if not text:
            continue
        
        # Remove e5 prefix
        if text.lower().startswith("passage:"):
            text = text[len("passage:"):].strip()
        
        if is_junk_content(text):
            continue
        
        tag = f"{result.metadata.get('source', 'Unknown')} (p.{result.metadata.get('page', '?')})"
        if tag in seen:
            continue
        
        seen.add(tag)
        context_chunks.append(text)
        sources.append(tag)
        
        if len(context_chunks) >= k:
            break
    
    if not context_chunks:
        return "", []
    
    return truncate_context("\n\n".join(context_chunks)), sources


# =========================
# Groq Interaction
# =========================
def call_groq_chat(messages: List[Dict[str, str]], max_tokens: int = 900) -> str:
    """Call Groq API with exponential backoff and error handling."""
    if not messages or not isinstance(messages, list):
        raise ValueError("messages must be a non-empty list")
    
    backoff = 1.0
    for attempt in range(1, 4):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle auth errors
            if any(x in error_str for x in ["401", "unauthorized", "invalid api"]):
                raise RuntimeError("Invalid GROQ API key or unauthorized access")
            
            # Handle model not found
            if "404" in error_str and "model" in error_str:
                raise RuntimeError(f"Model not found: {GROQ_MODEL}")
            
            # Handle rate limiting
            if any(x in error_str for x in ["429", "rate limit"]):
                raise RuntimeError("Rate limited. Please wait and retry.")
            
            # Retry on transient errors
            is_transient = any(x in error_str for x in 
                             ["timeout", "temporarily unavailable", "502", "503", "504"])
            
            if is_transient and attempt < 3:
                print(f"⚠️ Transient error (attempt {attempt}/3), retrying...")
                time.sleep(backoff)
                backoff *= 2
                continue
            
            raise
    
    raise RuntimeError("Groq request failed after retries")


# =========================
# Main Chatbot Function
# =========================
def get_chatbot_response(message: str, language: str = "English") -> Dict[str, Any]:
    """Generate agricultural response using RAG + Groq."""
    try:
        # Normalize inputs
        language = language.strip() if language else "English"
        if language not in ALLOWED_LANGS:
            language = "English"
        
        msg = (message or "").strip()
        
        if not msg:
            return {
                "response": "Please enter a question about your crop.",
                "source": None,
            }
        
        # Retrieve context
        context, sources = retrieve_context(msg, k=TOP_K)
        
        if not context:
            fallback = {
                "English": "I couldn't find relevant information. Please provide details about crop, growth stage, location, and symptoms.",
                "नेपाली": "सम्बन्धित जानकारी भेटिएन। कृपया बाली, विकास चरण, स्थान र लक्षणबारे विवरण दिनुहोस्।",
                "हिन्दी": "संबंधित जानकारी नहीं मिली। कृपया फसल, वृद्धि अवस्था, स्थान और लक्षणों के बारे में बताएं।",
            }
            return {"response": fallback.get(language, fallback["English"]), "source": None}
        
        # Build prompt
        system_prompt = (
            "You are an expert multilingual agricultural assistant. "
            "Use ONLY the provided context for facts. If context is insufficient, say so. "
            "Provide clear, numbered steps for advice."
        )
        
        user_prompt = f"""Language: {language}

Context:
{context}

Question:
{msg}

Answer in {language}:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        answer = call_groq_chat(messages)
        if not answer:
            answer = "Unable to generate response. Please try again."
        
        return {
            "response": answer,
            "source": "; ".join(sources) if sources else None,
        }
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Chatbot error: {type(e).__name__}: {e}")
        print(f"Traceback:\n{tb}")
        
        error_responses = {
            "rate limit": "Model is busy. Please try again later.",
            "api key": "Configuration error. Check your API key.",
            "not found": "Model not available.",
        }
        error_msg = str(e).lower()
        response = next((v for k, v in error_responses.items() if k in error_msg), 
                       "Error processing request. Please try again.")
        return {"response": response, "source": None}


# =========================
# Initialization
# =========================
def initialize_chatbot():
    """Initialize all global state."""
    global DEVICE, GROQ_MODEL, vector_store
    
    print("🚀 Initializing Agricultural Chatbot...")
    DEVICE = choose_device()
    GROQ_MODEL = pick_groq_model()
    
    try:
        vector_store = initialize_vectorstore()
        print("✅ Chatbot ready!")
    except RuntimeError as e:
        print(f"❌ Initialization failed: {e}")
        raise


# Call initialization
if __name__ != "__main__":
    initialize_chatbot()