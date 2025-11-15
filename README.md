MediBot – RAG Powered Medical Assistant

MediBot is a Retrieval-Augmented Generation (RAG) based medical chatbot that answers symptom-based and disease-related queries using medical PDFs.
It uses:

PDF → text extraction

Text cleaning & chunking

Embedding generation (Gemini/Gecko)

Pinecone vector database for similarity search

LLM for context-aware answers

🚀 Features

Upload medical PDFs as knowledge base

Query any symptom or disease

Accurate, document-grounded answers

Uses embeddings for similarity search

Fully integrated Streamlit frontend

🛠 Tech Stack

Python

Flask

Pinecone

Google Gemini / Llama / OpenAI

Sentence Transformers

PyPDF

🧠 RAG Pipeline

Extract text from PDF

Chunk into 300–500 token segments

Generate embeddings

Store in Pinecone

Query → embed → retrieve top-k chunks

LLM generates safe, context-grounded answer

▶ Run Locally
pip install -r requirements.txt
python app.py
