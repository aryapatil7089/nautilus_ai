import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 1. Initialize the LLM (Using Mixtral on Groq)
print("Initializing LLM...")
# Fails safely if API key is not set
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)

# 2. Re-initialize our Hybrid Search pipeline
print("Waking up Hybrid Search Indexes...")
loader = PyPDFDirectoryLoader("data")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
chunks = text_splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# --- GRAPH NODES ---

def security_node(state: MessagesState):
    print("-> [Security Agent] Scanning query...")
    last_message = state["messages"][-1].content.lower()
    restricted_terms = ["salary", "salaries", "board meeting", "password", "employee records"]
    
    if any(term in last_message for term in restricted_terms):
        return {"messages": [AIMessage(content="SECURITY ALERT: Unauthorized access attempt detected. Query rejected.")]}
    return {"messages": []}

def security_router(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "SECURITY ALERT" in last_message.content:
        return END
    return "supervisor_node"

def supervisor_node(state: MessagesState):
    print("-> [Supervisor Agent] Evaluating intent...")
    query = state["messages"][-1].content.lower()
    legal_keywords = ["law", "penalty", "ofac", "ais", "spoofing", "compliance", "sanction", "article", "fine"]
    
    if any(word in query for word in legal_keywords):
        return {"messages": [AIMessage(content="ROUTING_TO_SEARCH")]}
    else:
        return {"messages": [AIMessage(content="ROUTING_TO_CHAT")]}

def supervisor_router(state: MessagesState):
    last_message = state["messages"][-1].content
    if "ROUTING_TO_SEARCH" in last_message:
        return "hybrid_search_node"
    return "direct_chat_node"

def hybrid_search_node(state: MessagesState):
    print("-> [Hybrid Search Node] Retrieving legal texts and generating answer...")
    query = state["messages"][-2].content # The user's actual question
    
    # Run the hybrid search
    docs = ensemble_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Instruct the AI to act as a strict legal assistant
    system_prompt = f"""You are Nautilus, a strict maritime compliance AI. 
    Answer the user's question using ONLY the provided legal context below. Keep your response under 3 sentences.
    If exact penalty amounts are not explicitly stated, simply explain what the text DOES say about the consequences of AIS manipulation, disabling, or modification (e.g., enhanced due diligence, investigation). 
    Do NOT invent connections to unrelated articles about pollution or broadcasting.
    
    CONTEXT:
    {context}"""

    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"messages": [response]}

def direct_chat_node(state: MessagesState):
    print("-> [Direct Chat Node] Generating conversational response...")
    query = state["messages"][-2].content
    
    system_prompt = "You are Nautilus, a helpful maritime compliance AI assistant. Keep responses brief and professional."
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
    return {"messages": [response]}

# --- BUILD GRAPH ---
builder = StateGraph(MessagesState)

builder.add_node("security_node", security_node)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("hybrid_search_node", hybrid_search_node)
builder.add_node("direct_chat_node", direct_chat_node)

builder.add_edge(START, "security_node")
builder.add_conditional_edges("security_node", security_router)
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("hybrid_search_node", END)
builder.add_edge("direct_chat_node", END)

nautilus_app = builder.compile()

# --- RUN TESTS ---
if __name__ == "__main__":
    print("\n\n=== NAUTILUS SYSTEM ONLINE ===")
    
    print("\n--- TEST 1: The Casual Chat ---")
    casual_state = nautilus_app.invoke({"messages": [HumanMessage(content="Hello, who are you?")]})
    print(f"Final Answer: {casual_state['messages'][-1].content}\n")

    print("--- TEST 2: The Legal Query ---")
    legal_state = nautilus_app.invoke({"messages": [HumanMessage(content="What are the penalties for AIS spoofing?")]})
    print(f"Final Answer: {legal_state['messages'][-1].content}\n")
