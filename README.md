# FIle-Q-A-RAG-Chatbot
This code sets up a Streamlit-based file question-answer chatbot using LangChain to perform retrieval-augmented generation (RAG) from uploaded PDF documents. The app allows users to upload PDFs, processes those files into chunks, and then retrieves relevant information from the documents to answer user questions using OpenAI's GPT-3.5-turbo model.

# Features of the App
<ul>1. Upload multiple PDF files for content extraction.</ul>
<ul>2. The uploaded documents are split into smaller chunks for efficient processing.</ul>
<ul>3. Embeddings of document chunks are created using OpenAI embeddings.</ul>
<ul>4. A Vector Database is built using Chroma to store document embeddings.</ul>
<ul>5. Use GPT-3.5-turbo for answering questions based on the uploaded PDF documents.</ul>
<ul>6. Chat interface to interact with the chatbot and ask questions.</ul>
<ul>7. Real-time token streaming of chatbot responses.</ul>
<ul>8. Display of top 3 document sources used to generate answers.</ul>

## UI of the App
<img width="1710" alt="Screenshot 2024-11-04 at 5 53 43 AM" src="https://github.com/user-attachments/assets/bf5fc203-14ce-4e7c-8b61-4f7abe9630d9">
