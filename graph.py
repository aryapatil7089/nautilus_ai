import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 1. Initialize LLM
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set the GROQ_API_KEY environment variable.")

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

# 2. Build / Wake Up Hybrid Search Engine
print("Initializing Hybrid Retrieval Engine...")
loader = PyPDFDirectoryLoader("data", glob="**/*.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
    separators=["\n\nSection", "\n\nRegulation", "\n\nArticle", "\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(documents)

# Sparse BM25
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 2

# Dense ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# --- GRAPH NODES ---

def security_node(state: MessagesState):
    last_message = state["messages"][-1].content.lower()
    restricted_terms = ["salary", "salaries", "board meeting", "password", "employee records"]
    
    if any(term in last_message for term in restricted_terms):
        return {"messages": [AIMessage(content="[SECURITY_BLOCK] I am a maritime compliance assistant. I am not authorized to discuss internal corporate data.")]}
    return {"messages": []}

def security_router(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "[SECURITY_BLOCK]" in last_message.content:
        return END
    return "supervisor_node"

def supervisor_node(state: MessagesState):
    history_summary = "\n".join([f"{m.type}: {m.content[:200]}" for m in state["messages"][-4:]])
    
    supervisor_prompt = f"""You are the master intent classifier for Nautilus Maritime AI.
Analyze the conversation history and the latest user query.

CONVERSATION HISTORY:
{history_summary}

Classify the user intent into EXACTLY ONE category:
- ACTIONABLE_INTERDICTION: The query asks for legal powers, boarding rights, sections, arrest, detention, fines, or reports on a specific suspect ship scenario.
- GENERAL_QUERY: The query is a general explanation (e.g. "what is dark fleet"), a greeting (e.g. "hi"), or a follow-up to previous chat (e.g. "explain simply", "summarize").
- IRRELEVANT: The query is completely unrelated to maritime topics, shipping, law, or security (e.g. cooking, coding, sports, weather).

Respond with ONLY the category name."""

    decision = llm.invoke([SystemMessage(content=supervisor_prompt)]).content.strip()
    
    if "ACTIONABLE_INTERDICTION" in decision:
        return {"messages": [AIMessage(content="ROUTE_TO_TACTICAL")]}
    elif "IRRELEVANT" in decision:
        return {"messages": [AIMessage(content="ROUTE_TO_REFUSAL")]}
    else:
        return {"messages": [AIMessage(content="ROUTE_TO_CHAT")]}

def supervisor_router(state: MessagesState):
    decision = state["messages"][-1].content
    if "ROUTE_TO_TACTICAL" in decision:
        return "tactical_enforcement_node"
    elif "ROUTE_TO_REFUSAL" in decision:
        return "refusal_node"
    return "conversational_maritime_node"

def tactical_enforcement_node(state: MessagesState):
    user_query = state["messages"][-2].content
    docs = ensemble_retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""You are Nautilus, an operational legal decision-support system for maritime compliance officers.
Analyze the user's operational query using the provided context and your foundational knowledge of international maritime treaties.

FORMATTING REQUIREMENTS:
- DO NOT use Markdown tables. Tables are strictly forbidden.
- Use clear Markdown headings and bullet points.
- Explain archaic maritime terminology in simple terms (e.g., 'national colours (ship's flag)').

STRUCTURE:
### 1. International Authorization
(Cite UNCLOS, SOLAS, MARPOL, or IMO Resolutions establishing baseline jurisdiction)

### 2. Domestic Statutory Violation
(Cite specific sections under Indian Merchant Shipping legislation and associated 2026 Rules)

### 3. Actionable Enforcement & Penalties
(List concrete powers: boarding authority, detention, port denial, seizure, and statutory fine amounts)

CONTEXT:
{context}"""

    conversation_payload = [SystemMessage(content=system_prompt)] + state["messages"][:-1]
    response = llm.invoke(conversation_payload)
    return {"messages": [response]}

def conversational_maritime_node(state: MessagesState):
    user_query = state["messages"][-2].content
    docs = ensemble_retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""You are Nautilus, an expert AI assistant specializing in maritime compliance and dark fleet interdiction.
Help the user with their greeting, question, or follow-up request in a professional, clear, and direct manner.

GUIDELINES:
- If the user asks for a simple or short explanation, deliver a concise summary without technical tables or rigid legal structures.
- If this is a follow-up to a previous question, maintain conversational context and explain the earlier concepts as requested.
- If the user is greeting you, introduce yourself and state your purpose.

LEGAL REFERENCE CONTEXT:
{context}"""

    conversation_payload = [SystemMessage(content=system_prompt)] + state["messages"][:-1]
    response = llm.invoke(conversation_payload)
    return {"messages": [response]}

def refusal_node(state: MessagesState):
    msg = AIMessage(content="I am Nautilus, a specialized maritime regulatory and compliance system. I can only assist with maritime law, vessel tracking, sanctions enforcement, and dark fleet compliance.")
    return {"messages": [msg]}

# --- BUILD STATE GRAPH ---
builder = StateGraph(MessagesState)

builder.add_node("security_node", security_node)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("tactical_enforcement_node", tactical_enforcement_node)
builder.add_node("conversational_maritime_node", conversational_maritime_node)
builder.add_node("refusal_node", refusal_node)

builder.add_edge(START, "security_node")
builder.add_conditional_edges("security_node", security_router)
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("tactical_enforcement_node", END)
builder.add_edge("conversational_maritime_node", END)
builder.add_edge("refusal_node", END)

nautilus_app = builder.compile()
