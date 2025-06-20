import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def load_models_and_client():
    """Load all models and initialize Qdrant client"""
    try:
        # Get API keys
        google_api_key = os.getenv("GOOGLE_API_KEY")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not google_api_key or not qdrant_api_key:
            st.error("Please set GOOGLE_API_KEY and QDRANT_API_KEY in your .env file")
            return None
        
        # Initialize embedding models
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        
        # Initialize Qdrant client
        client = QdrantClient(
            url="https://3954ad4f-f6a9-4e8a-9c7d-7d00cfe00fe9.us-east4-0.gcp.cloud.qdrant.io",
            api_key=qdrant_api_key
        )
        
        # Initialize chat model
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            max_output_tokens=1000,
            disable_streaming=False,
            api_key=google_api_key
        )
        
        return {
            "sentence_model": sentence_model,
            "dense_embedding_model": dense_embedding_model,
            "bm25_embedding_model": bm25_embedding_model,
            "late_interaction_embedding_model": late_interaction_embedding_model,
            "client": client,
            "chat_model": chat_model
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def search_qdrant(query, models_dict, limit=3):
    """Search the Qdrant collection and return documents"""
    try:
        # Generate embeddings for the query using all three models
        query_dense = models_dict["sentence_model"].encode(query).tolist()
        query_sparse = next(models_dict["bm25_embedding_model"].query_embed(query))
        query_late = next(models_dict["late_interaction_embedding_model"].query_embed(query))
        
        # Search Qdrant collection
        results = models_dict["client"].query_points(
            collection_name="hybrid-search4",
            query=query_late,
            using="colbertv2.0",
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="all-MiniLM-L6-v2",
                    limit=limit
                ),
                models.Prefetch(
                    query=models.SparseVector(**query_sparse.as_object()),
                    using="bm25",
                    limit=limit
                )
            ],
            with_payload=True,
            limit=limit
        )
        
        # Convert results to Document objects
        documents = []
        for result in results.points:
            documents.append(
                Document(
                    page_content=result.payload["document"],
                    metadata={"id": result.id, "score": result.score}
                )
            )
        
        return documents
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def rag_chat(question, models_dict, k=3):
    """RAG chatbot function"""
    try:
        # Step 1: Retrieve relevant documents
        relevant_docs = search_qdrant(question, models_dict, limit=k)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "context_length": 0
            }
        
        # Step 2: Combine retrieved content
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Step 3: Create messages
        system_message = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
If you can't find the answer in the context, say so clearly.

Context:
{context}"""
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ]
        
        # Step 4: Get response from chat model
        response = models_dict["chat_model"].invoke(messages)
        
        return {
            "answer": response.content,
            "sources": relevant_docs,
            "context_length": len(context)
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": [],
            "context_length": 0
        }

# Main app
def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about the CNN/DailyMail articles using hybrid search!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Number of documents to retrieve
        k_docs = st.slider("Number of documents to retrieve", 1, 10, 3)
        
        # Search method selection
        search_method = st.selectbox(
            "Search Method",
            ["Hybrid (ColBERT + Dense + Sparse)", "Dense Only", "Sparse Only"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot uses:
        - **Dense embeddings**: all-MiniLM-L6-v2
        - **Sparse embeddings**: BM25
        - **Late interaction**: ColBERT v2.0
        - **LLM**: Google Gemini 2.0 Flash
        """)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models and connecting to Qdrant..."):
            models_dict = load_models_and_client()
            if models_dict:
                st.session_state.models_dict = models_dict
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please check your configuration.")
                return
    
    # Chat interface
    if st.session_state.models_loaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}** (Score: {doc.metadata['score']:.4f})")
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the articles..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating response..."):
                    result = rag_chat(prompt, st.session_state.models_dict, k=k_docs)
                    
                    st.markdown(result["answer"])
                    
                    # Show sources
                    if result["sources"]:
                        with st.expander("📄 Sources"):
                            for i, doc in enumerate(result["sources"]):
                                st.markdown(f"**Source {i+1}** (Score: {doc.metadata['score']:.4f})")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
