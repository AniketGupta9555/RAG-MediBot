Check demo: https://www.loom.com/share/e6c6e80663ad40aea742093630e0a28b
MediBot – RAG Powered Medical Assistant

MediBot is a Retrieval-Augmented Generation (RAG) medical assistant built using Flask.
It answers symptom-based and disease-related queries using medical PDFs, embeddings, and a vector database.

🚀 Features

Upload medical PDFs to build the knowledge base

Extract, clean, and chunk text

Generate embeddings and store them in a vector DB (Pinecone/FAISS)

Retrieve top-k relevant chunks for any query

LLM generates accurate, context-grounded medical responses

Safe, fast, and explainable medical assistant

🛠 Tech Stack

Flask

Sentence Transformers / Gemini Embeddings

Pinecone or FAISS

PyPDF

Python-dotenv

▶ Run Locally
pip install -r requirements.txt
python app.py
