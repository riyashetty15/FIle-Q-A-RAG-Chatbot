# FIle-QA-RAG-Chatbot
This code sets up a Streamlit-based file question-answer chatbot using LangChain to perform retrieval-augmented generation (RAG) from uploaded PDF documents. The app allows users to upload PDFs, processes those files into chunks, and then retrieves relevant information from the documents(papers) to answer user questions using OpenAI's GPT-3.5-turbo model.

# Features of the App
<ul>1. Upload multiple PDF files for content extraction.</ul>
<ul>2. The uploaded documents are split into smaller chunks for efficient processing.</ul>
<ul>3. Embeddings of document chunks are created using OpenAI embeddings.</ul>
<ul>4. A Vector Database is built using Chroma to store document embeddings.</ul>
<ul>5. Use GPT-3.5-turbo for answering questions based on the uploaded PDF documents.</ul>
<ul>6. Chat interface to interact with the chatbot and ask questions.</ul>
<ul>7. Real-time token streaming of chatbot responses.</ul>
<ul>8. Display of top 3 document sources used to generate answers.</ul>

# Functioning of the App
<ul><b>Upload PDFs:</b> Users upload one or more PDF files via the app's sidebar.</ul>
<ul><b>Document Processing:</b>
The app reads and processes the uploaded PDFs. Each document is split into smaller text chunks using LangChain's RecursiveCharacterTextSplitter to handle large documents efficiently. Document chunks are converted into embeddings using OpenAIEmbeddings. The embeddings and document chunks are stored in a Vector Database powered by Chroma.</ul>

<ul>Once files are uploaded, they are processed using PyMuPDFLoader from LangChain to extract the content. The content is split into smaller text chunks using RecursiveCharacterTextSplitter. This ensures that documents are processed efficiently and can handle large files by breaking them into chunks of around 1500 characters with a 200-character overlap.</ul>

<ul>Document chunks are embedded into vector representations using OpenAIEmbeddings. The vector embeddings are stored in a Chroma vector database. A retriever is created from the vector database that will retrieve the most relevant chunks based on user queries.</ul>

<ul><b>Retriever:</b> A retriever is created to query the vector database based on user questions. This retriever looks for relevant document chunks that contain the context necessary to answer the user's question.</ul>

<ul><b>Question-Answering (QA):</b> A user enters a question via the chat interface. The retriever fetches relevant chunks from the Vector Database. A prompt template combines the retrieved chunks and the user's question. The prompt is passed to GPT-3.5-turbo to generate an answer. The answer is displayed in the chat interface, with live token streaming for real-time feedback.</ul>
  
<ul><b>Source Display:</b> After the answer is generated, the app displays the top 3 document sources the model uses to answer the question. The app uses a PostMessageHandler to extract and display the top 3 document sources used to answer the user's question. This information is displayed in a Pandas data frame. The chat history is managed using StreamlitChatMessageHistory, which stores and displays both user inputs and AI-generated responses.</ul>

## UI of the App
<img width="1710" alt="Screenshot 2024-11-04 at 5 53 43 AM" src="https://github.com/user-attachments/assets/bf5fc203-14ce-4e7c-8b61-4f7abe9630d9">
