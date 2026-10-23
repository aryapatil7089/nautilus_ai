import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from graph import nautilus_app

# Configure application metadata and widescreen view
st.set_page_config(
    page_title="Nautilus Maritime Compliance Hub",
    page_icon="⚓",
    layout="wide"
)

st.title("⚓ Nautilus: Multi-Agent Maritime Compliance Hub")
st.caption("Secure-by-Design Multi-Agent Hybrid Retrieval Pipeline | OFAC • ISPS • UNCLOS")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Displays real-time architecture telemetry for reviewers
with st.sidebar:
    st.markdown("### 🛡️ Multi-Agent Telemetry")
    st.markdown("**Active Architecture Controls:**")
    st.markdown("- **Security Layer:** Lexical screening & query boundary enforcement")
    st.markdown("- **Supervisor Node:** Semantic routing & intent classification")
    st.markdown("- **Retrieval Engine:** Hybrid BM25 (Sparse) + ChromaDB (Dense) via RRF")

# Render historical chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Enter a maritime regulatory or compliance query..."):
    # Append user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Convert Streamlit history to LangChain message objects
    messages_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_history.append(AIMessage(content=msg["content"]))

    # Generate response using the graph
    with st.chat_message("assistant"):
        with st.spinner("Processing multi-agent routing graph..."):
            try:
                output = nautilus_app.invoke({"messages": messages_history})
                response_content = output["messages"][-1].content
                st.markdown(response_content)
                # Append assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
