from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

print("Waking up Nautilus Search Engine...")

# 1. Re-load chunks to build the BM25 Keyword Index (Sparse)
loader = PyPDFDirectoryLoader("data")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
chunks = text_splitter.split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3 # Retrieve top 3 keyword matches

# 2. Load the Chroma Vector Index we just built (Dense)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
chroma_retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 semantic matches

# 3. Combine them using EnsembleRetriever (This automatically handles Reciprocal Rank Fusion!)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5] # Give 50% importance to exact keywords, 50% to meaning
)

# 4. Test the Hybrid Search
query = "What are the penalties or warnings regarding AIS spoofing and deceptive practices?"
print(f"\nSearching for: '{query}'\n")

results = ensemble_retriever.invoke(query)

print("--- TOP 3 FUSED RESULTS ---")
# We only print the top 3 best paragraphs the system found
for i, doc in enumerate(results[:3]):
    # Get the file name so we know where the rule came from
    source = doc.metadata.get('source', 'Unknown file').split('/')[-1]
    print(f"\nResult {i+1} (Found in {source}):")
    print(doc.page_content)
    print("-" * 50)