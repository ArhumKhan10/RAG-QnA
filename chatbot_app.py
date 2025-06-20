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
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
if "metrics" not in st.session_state:
    st.session_state.metrics = []

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
    """RAG chatbot function with metrics tracking"""
    start_time = time.time()
    
    try:
        # Retrieve relevant documents
        retrieval_start = time.time()
        relevant_docs = search_qdrant(question, models_dict, limit=k)
        retrieval_time = time.time() - retrieval_start
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "context_length": 0,
                "metrics": {
                    "total_time": time.time() - start_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "num_tokens_prompt": 0,
                    "num_tokens_completion": 0,
                    "similarity_scores": []
                }
            }
        
        # Combine retrieved content
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create messages
        system_message = f"""You are a helpful AI assistant. Use only the provided context from news articles to answer the user's question as accurately as possible. 
                            If the answer cannot be found in the context, clearly state that you do not know. Do not make up information or use outside knowledge. 
                            Always cite relevant sources from the context when possible.

Context:
{context}"""
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ]
        
        # Count prompt tokens (approximation)
        prompt_text = system_message + question
        prompt_tokens = len(prompt_text.split())
        
        # Step 4: Get response from chat model
        generation_start = time.time()
        response = models_dict["chat_model"].invoke(messages)
        generation_time = time.time() - generation_start
        
        # Count completion tokens (approximation)
        completion_tokens = len(response.content.split())
        
        # Extract similarity scores
        similarity_scores = [doc.metadata['score'] for doc in relevant_docs]
        
        total_time = time.time() - start_time
        
        # Update metrics in session state
        st.session_state.metrics.append({
            "question": question,
            "total_time": total_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "num_tokens_prompt": prompt_tokens,
            "num_tokens_completion": completion_tokens,
            "similarity_scores": similarity_scores,
            "timestamp": datetime.now()
        })
        
        return {
            "answer": response.content,
            "sources": relevant_docs,
            "context_length": len(context),
            "metrics": {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "num_tokens_prompt": prompt_tokens,
                "num_tokens_completion": completion_tokens,
                "similarity_scores": similarity_scores,
                "timestamp": datetime.now()
            }
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": [],
            "context_length": 0,
            "metrics": {
                "total_time": time.time() - start_time,
                "retrieval_time": 0,
                "generation_time": 0,
                "num_tokens_prompt": 0,
                "num_tokens_completion": 0,
                "similarity_scores": [],
                "timestamp": datetime.now()
            }
        }

def show_metrics_dashboard():
    """Display metrics dashboard"""
    st.header("📊 Metrics Dashboard")
    
    if not st.session_state.metrics:
        st.info("No metrics available yet. Start chatting to see metrics!")
        return
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(st.session_state.metrics)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_total_time = metrics_df['total_time'].mean()
        st.metric("Avg Response Time", f"{avg_total_time:.2f}s")
    
    with col2:
        avg_retrieval_time = metrics_df['retrieval_time'].mean()
        st.metric("Avg Retrieval Time", f"{avg_retrieval_time:.2f}s")
    
    with col3:
        avg_generation_time = metrics_df['generation_time'].mean()
        st.metric("Avg Generation Time", f"{avg_generation_time:.2f}s")
    
    with col4:
        total_queries = len(metrics_df)
        st.metric("Total Queries", total_queries)
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        # Response time over time
        query_numbers = list(range(len(metrics_df)))
        fig_time = px.line(metrics_df, x=query_numbers, y='total_time', 
                          title="Response Time Over Queries", 
                          labels={'x': 'Query Number', 'y': 'Response Time (s)'})
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Token usage
        fig_tokens = go.Figure()
        query_numbers = list(range(len(metrics_df)))
        fig_tokens.add_trace(go.Bar(name='Prompt Tokens', x=query_numbers, y=metrics_df['num_tokens_prompt']))
        fig_tokens.add_trace(go.Bar(name='Completion Tokens', x=query_numbers, y=metrics_df['num_tokens_completion']))
        fig_tokens.update_layout(title="Token Usage per Query", barmode='stack')
        st.plotly_chart(fig_tokens, use_container_width=True)
    
    # Similarity scores distribution
    all_scores = [score for scores in metrics_df['similarity_scores'] for score in scores]
    if all_scores:
        fig_scores = px.histogram(x=all_scores, nbins=20, title="Distribution of Similarity Scores")
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Metrics")
    display_df = metrics_df.copy()
    display_df['similarity_scores'] = display_df['similarity_scores'].apply(lambda x: f"{len(x)} scores" if x else "No scores")
    st.dataframe(display_df)

# Main app
def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about the CNN/DailyMail articles using hybrid search!")
    
    # Add tabs for chat and metrics
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Metrics"])
    
    with tab1:
        # Sidebar
        with st.sidebar:
            st.header("Settings")
              # Number of documents to retrieve
            k_docs = st.slider("Number of documents to retrieve", 1, 10, 3, key="k_docs_slider")
            
            # Search method selection
            search_method = st.selectbox(
                "Search Method",
                ["Hybrid (ColBERT + Dense + Sparse)", "Dense Only", "Sparse Only"],
                key="search_method_select"
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
                        
                        # Store metrics
                        if "metrics" in result:
                            st.session_state.metrics.append(result["metrics"])
                        
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
            if st.button("Clear Chat", key="clear_chat_btn"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        show_metrics_dashboard()

if __name__ == "__main__":
    main()
