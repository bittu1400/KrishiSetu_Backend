# create_metadata.py
import pickle
from pathlib import Path
from langchain.docstore.document import Document
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from typing import List

# =========================
# Configuration
# =========================
DATA_DIR = Path("RAG/data")  # Changed to Path object
METADATA_PATH = Path("vector_store/faiss_index/metadata.pkl")

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
        
        # Try PyPDF first
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
            print(f"⚠️ PyPDF error on {file_path.name}: {e}")
        
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
# Main Execution
# =========================
if __name__ == "__main__":
    print("="*60)
    print("Creating Metadata from PDFs")
    print("="*60)
    
    # Load documents
    print(f"\n📂 Loading PDFs from {DATA_DIR}...")
    your_original_documents = load_pdfs_from_directory(DATA_DIR)
    
    if not your_original_documents:
        print("❌ No documents loaded. Exiting.")
        exit(1)
    
    print(f"✅ Loaded {len(your_original_documents)} document pages")
    
    # Create metadata
    print("\n🔧 Creating metadata...")
    documents_metadata = []
    
    for doc in your_original_documents:
        documents_metadata.append({
            'text': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', '?')
        })
    
    # Save metadata
    print(f"\n💾 Saving metadata to {METADATA_PATH}...")
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(documents_metadata, f)
    
    print(f"✅ Saved metadata for {len(documents_metadata)} documents")
    
    # Verify file size
    size_mb = METADATA_PATH.stat().st_size / (1024 * 1024)
    print(f"📦 Metadata file size: {size_mb:.2f} MB")
    
    print("="*60)
    print("✅ Metadata creation complete!")
    print("="*60)