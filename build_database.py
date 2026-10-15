from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Step 1: Loading PDFs from the data folder...")
loader = PyPDFDirectoryLoader("data")
documents = loader.load()
print(f"Success! Loaded {len(documents)} total pages.")

print("\nStep 2: Chunking documents to preserve legal context...")
# We use 500 characters with a 100-character overlap as per the project requirements
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=250
)
chunks = text_splitter.split_documents(documents)
print(f"Success! Split the pages into {len(chunks)} overlapping chunks.")

print("\nStep 3: Converting text to embeddings and building the Vector Database...")
# We use a free, lightweight open-source embedding model so you don't need paid API keys
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# This creates a folder called 'chroma_db' to permanently save our data
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Database built and saved successfully! Nautilus now has a memory.")