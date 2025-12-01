This project is a small RAG chatbot system with a Flask backend and a clean HTML/JS frontend.
You can upload a PDF to build the knowledge base, and then ask questions through the chatbot interface. The backend extracts the text, creates chunks, generates vector embeddings, and stores them in a FAISS database. The chatbot retrieves the most relevant chunks and returns an accurate answer with context.

Main steps:

Upload PDF → backend extracts text
Text → split into chunks
Chunks → create embeddings
Embeddings → stored in FAISS vector DB
Ask question → retrieve chunks → generate answer → show on UI
