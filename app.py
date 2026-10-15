import streamlit as st
from langchain_core.messages import HumanMessage
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
    st.header("🛡️ Multi-Agent Telemetry")
    st.markdown("**Active Architecture Controls:**")
    st.markdown("- **Security Layer:** Lexical screening & query boundary enforcement")
    st.markdown("- **Supervisor Node:** Semantic routing & intent classification")
    st.markdown("- **Retrieval Engine:** Hybrid BM25 (Sparse) + ChromaDB (Dense) via RRF")
    st.markdown("---")
    st.markdown("### Execution Trace")
    log_placeholder = st.empty()
    log_placeholder.info("Ready for input. Waiting for user query...")

# Render historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new user input
if prompt := st.chat_input("Enter a maritime regulatory or compliance query..."):
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the LangGraph multi-agent state machine
    with st.chat_message("assistant"):
        with st.spinner("Traversing multi-agent routing graph..."):
            try:
                state_output = nautilus_app.invoke({
                    "messages": [HumanMessage(content=prompt)]
                })
                final_response = state_output["messages"][-1].content
                
                # Check for the hidden security tag
                if "[SECURITY_BLOCK]" in final_response:
                    # Remove the tag so the user just sees the polite refusal
                    clean_response = final_response.replace("[SECURITY_BLOCK] ", "")
                    st.markdown(clean_response)
                    log_placeholder.warning("⚠️ **Security Node Triggered:** Boundary enforcement activated. Pipeline halted.")
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
                else:
                    st.markdown(final_response)
                    log_placeholder.success("✅ **Pipeline Execution:** Security Cleared → Supervisor Dispatched → Context Fused via RRF")
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"Pipeline Execution Error: {str(e)}")
