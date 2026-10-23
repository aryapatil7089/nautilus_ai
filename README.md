# ⚓ Nautilus AI: Multi-Agent Maritime Compliance Hub

**Author:** Arya Patil  
**Institution:** Imarticus Learning (Postgraduate Program in Data Science & Analytics)  
**Track:** Applied Generative AI & Data Analytics Portfolio Project

---

### 🚀 Live Web Application
**[👉 CLICK HERE TO ACCESS THE LIVE NAUTILUS PORTAL](https://nautilusai-sg6babyw6atdsqp9nxpi5c.streamlit.app/)**

> **⚠️ IMPORTANT CLOUD HOSTING NOTICE:**  
> This application is deployed on Streamlit Community Cloud. If the application has not been accessed recently, the container may be asleep. Please allow 15–30 seconds for the cloud instance to wake up, load the vector embeddings, and boot the multi-agent graph.

---

## 🚢 Problem Statement
During critical maritime interdictions (e.g., stopping a dark fleet vessel, boarding a suspected smuggler), coast guard and compliance officers operate under extreme time constraints. Searching through thousands of pages of dense international treaties (UNCLOS, SOLAS, MARPOL) and localized domestic statutes to determine legal boarding authority or detention penalties is inefficient. A highly accurate, hallucination-free legal retrieval system is required to translate complex maritime law into instant, actionable tactical briefings.

## 🚀 Overall Solution Approach
Nautilus AI is a Secure-by-Design Multi-Agent Retrieval-Augmented Generation (RAG) system. Rather than relying on a single LLM prompt, the architecture utilizes **LangGraph** to construct a deterministic state machine, routing user queries through specialized AI agents to ensure strict compliance and formatting.

* **High-Speed Inference:** Powered by the **Groq API** utilizing the highly efficient `openai/gpt-oss-20b` model for rapid, low-latency reasoning.
* **Intelligent Semantic Routing:** A Supervisor Agent evaluates the user's intent based on conversation history and routes the query to either a Tactical Enforcement Node, a Conversational Node, or a Security Refusal Node.
* **Hybrid Search Engine (RRF):** To guarantee high-precision legal retrieval, the system fuses two search methodologies:
  * **Sparse Retrieval (BM25):** Excels at exact keyword matches (e.g., "Section 12(1)", "MMSI").
  * **Dense Retrieval (ChromaDB + HuggingFace `all-MiniLM-L6-v2`):** Excels at semantic context (e.g., "what happens if a ship hides its identity").
  * The results are combined using **Reciprocal Rank Fusion (RRF)** for optimal document retrieval.
* **Strict Lexical Security:** A boundary enforcement layer automatically intercepts and blocks non-maritime queries or attempts to extract internal corporate data.

## 📚 Legal Data Framework
The underlying vector database is built upon a curated repository of 9 critical maritime law documents:
* **International Treaties:** UNCLOS (Law of the Sea), SOLAS (Safety of Life at Sea), MARPOL (Marine Pollution), ISPS Code, and OFAC Sanctions guidelines.
* **Domestic Statutes:** Indian Merchant Shipping legislation and associated territorial enforcement rules.

## 📊 Key Findings & Architectural Decisions
Building legal-grade AI requires prioritizing deterministic outputs over creative text generation.
* **Preventing Hallucinated Formatting:** Standard LLMs tend to generate unpredictable Markdown tables that break UI rendering. We engineered a strict prompt template that completely forbids HTML `<br>` tags and tables, forcing the LLM to output a standardized 3-step briefing: *International Authorization, Domestic Violation, and Actionable Enforcement.*
* **Decoupling from Live ML:** Initially considered as a unified architecture with a real-time K-Means/DBSCAN radar, Nautilus AI was decoupled into an independent Streamlit microservice. This isolates the heavy NLP vector memory from the live WebSocket telemetry memory, preventing free-tier cloud server crashes.
* **Pre-Computed Embeddings:** To ensure instant cloud booting, the ChromaDB vector embeddings were pre-built locally and pushed directly to the repository, entirely bypassing the need to re-parse PDFs during server startup.

## 💻 UI Features
The frontend is built on **Streamlit** and designed for operational clarity:
* **Multi-Agent Telemetry Sidebar:** Displays the active architecture controls (Security Layer, Supervisor Node, Retrieval Engine) so reviewers can see the invisible graph mechanics operating behind the chat.
* **Persistent Chat Memory:** The system retains conversational state, allowing users to ask natural follow-up questions (e.g., *"Summarize that last point more simply"*).

## 📁 Repository Structure
* [`app.py`](./app.py): The Streamlit frontend, UI configuration, and session state management.
* [`graph.py`](./graph.py): The core LangGraph state machine, Agent definitions, Hybrid Retriever setup, and Groq LLM integration.
* [`build_database.py`](./build_database.py): Offline ingestion script to parse PDFs, chunk text (RecursiveCharacterTextSplitter), and generate embeddings.
* [`test_search.py`](./test_search.py): Diagnostics script to validate BM25 and ChromaDB vector retrieval accuracy.
* [`requirements.txt`](./requirements.txt): Pinned Python dependencies optimized for Streamlit Cloud stability.
* [`chroma_db/`](./chroma_db/): The pre-compiled, persistent dense vector database.
* [`data/`](./data/): The directory containing the source maritime PDFs.
* [`.gitignore`](./.gitignore): Standard Git exclusion rules.

## 🔮 Future Scope
* **Live API Integration:** Introduce functional Tool Calling allowing the Tactical Agent to query the Dark Fleet AI Monitor's SQLite database to autonomously fetch a suspect vessel's anomaly score and cross-reference it with the legal database.
* **Multi-Modal Document Parsing:** Upgrade the PDF ingestion pipeline to process unstructured legal images and scanned maritime registry documents using OCR.

## ⚙️ How to Run the Project Locally

1. Clone this repository to your local machine.
2. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
3. Export your free Groq API key: 
   * Mac/Linux: `export GROQ_API_KEY="your_key_here"`
   * Windows: `set GROQ_API_KEY="your_key_here"`
4. Run the Streamlit interface: `streamlit run app.py`
5. Open your browser and navigate to `http://localhost:8501`.
