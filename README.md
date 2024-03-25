# DocsBot-AI
Create your own Chatbot (ChatGPT) with your documents using Langchain and Gradio. 
This also uses:
- HuggingFaceEmbeddings for embeddings
- ChromaDB for a vectorstore
- OpenAI for a text generation model

# High level architecture of Chatbot
![High level architecture of Chatbot](images/architecture.png)

# How it works
- ```ingest.py``` uses LangChain tools to parse the document and create embeddings ```HuggingFaceEmbeddings```. It then stores the result in a vector database using ```Chroma``` vector store.
- ```myGPT.py``` using ChatOpenAI understand questions and create answers. The context for the answers is extracted from the vector store using a similarity search to locate the right piece of context from the docs.

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```
pip install -r requirements.txt
```


