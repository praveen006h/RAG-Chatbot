# RAG-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, HuggingFace Embeddings, and Groq.

This project allows you to load local documents into a vector database and query them using an advanced LLM, ensuring that the model's answers are grounded in your own data.

## Features

- **Document Loading & Chunking**: Ingest documents from a local `data/` directory.
- **Local Vector Database**: Store and retrieve document embeddings using ChromaDB.
- **Open-Source Embeddings**: Uses `BAAI/bge-small-en-v1.5` from HuggingFace for creating embeddings (with GPU support if available).
- **Fast LLM Inference**: Leverages Groq's API to run the `llama-3.1-8b-instant` model for generating responses based on the retrieved context.

## Prerequisites

- Python 3.12+
- A [Groq API Key](https://console.groq.com/keys)

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd RAG-Chatbot
   ```

2. **Set up the environment**:
   Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Install dependencies**:
   It is recommended to use a virtual environment or `uv`.
   ```bash
   pip install -r requirements.txt
   # or if using uv
   uv sync
   ```

## Usage

### 1. Ingest Data

Place your source documents (e.g., text files, markdown) into the `data/` directory.

Run the ingestion script to parse the documents, chunk them, compute embeddings, and store them in the `chroma/` database:

```bash
python create_vec_db.py
```

### 2. Query the Chatbot

Use the `query_data.py` script to ask questions about your documents:

```bash
python query_data.py "Your question here?"
```

The script will search the vector database for the most relevant context and use the Groq LLM to answer the question, providing the sources it used.
