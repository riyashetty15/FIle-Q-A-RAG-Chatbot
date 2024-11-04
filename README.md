# FIle-Q-A-RAG-Chatbot
This code sets up a Streamlit-based file question-answer chatbot using LangChain to perform retrieval-augmented generation (RAG) from uploaded PDF documents. The app allows users to upload PDFs, processes those files into chunks, and then retrieves relevant information from the documents to answer user questions using OpenAI's GPT-3.5-turbo model.

# Features of the App
<ul>1. Upload multiple PDF files for content extraction.</ul>
<li>2. The uploaded documents are split into smaller chunks for efficient processing.</li>
<li>3. Embeddings of document chunks are created using OpenAI embeddings.</li>
<li>4. A Vector Database is built using Chroma to store document embeddings.</li>
<li>5. Use GPT-3.5-turbo for answering questions based on the uploaded PDF documents.</li>
<li>6. Chat interface to interact with the chatbot and ask questions.</li>
<li>7. Real-time token streaming of chatbot responses.</li>
<li>8. Display of top 3 document sources used to generate answers.</li>
