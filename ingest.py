import os
import glob
from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"})
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str):
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]

def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    chunk_size = 500
    chunk_overlap = 50
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, 
                               persist_directory=persist_directory, 
                               client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    main()