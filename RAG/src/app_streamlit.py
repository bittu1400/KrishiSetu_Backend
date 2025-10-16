import streamlit as st
import os
from rag_chain import create_rag_chain

# Page config
st.set_page_config(
    page_title="Free RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_rag():
    """Initialize the RAG chain."""
    try:
        if st.session_state.rag_chain is None:
            with st.spinner("🔧 Initializing RAG system..."):
                st.session_state.rag_chain = create_rag_chain()
        return True
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return False

def main():
    st.title("🤖 Free LangChain RAG Chatbot")
    st.markdown("Ask questions about your uploaded documents!")
    
    # Sidebar with info
    with st.sidebar:
        st.header("📋 Information")
        st.markdown("""
        **How it works:**
        1. Documents are stored in `data/raw/`
        2. Run `ingest_free.py` to index them
        3. Ask questions here!
        
        **Tech Stack:**
        - 🤖 GPT4All (Local LLM)
        - 🔮 HuggingFace Embeddings
        - 🗄️ ChromaDB
        - 🚀 Streamlit UI
        """)
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize RAG system
    if not initialize_rag():
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.text(source)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.rag_chain.query(prompt)
                
                st.markdown(result["answer"])
                
                # Add sources
                if result["sources"]:
                    with st.expander("📚 Sources"):
                        for source in result["sources"]:
                            st.text(source)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

if __name__ == "__main__":
    main()
