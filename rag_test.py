import os
from dotenv     import load_dotenv

# Load environment variables from .env file
load_dotenv()
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


def process_pdf(pdf_path):
    """Reads the PDF, splits text into chunks, and creates embeddings."""
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    # Split text into manageable chunks (500 characters each)
    chunk_size = 500
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    return chunks

def setup_database(chunks):
    """Stores the text chunks and their embeddings into ChromaDB."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    
    print("Generating embeddings...")
    embeddings = model.encode(chunks)
    
    # Initialize in-memory ChromaDB
    client = chromadb.Client()
    
    # Delete collection if it already exists (useful for testing)
    try:
        client.delete_collection("my_knowledge_base")
    except:
        pass
    
    collection = client.create_collection("my_knowledge_base")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection, model

def get_answer(query, collection, model):
    """Retrieves relevant context and asks the AI to answer based on it."""
    
    # 1. Find the most relevant chunks from the database
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=2)
    
    # Combine the top 2 chunks to give AI better context
    context = "\n\n".join(results['documents'][0])
    
    # 2. Ask the AI
    client = Groq()
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer is not there, say 'I don't know'. Keep the answer short and in English."
            },
            {"role": "user", "content": prompt}
        ],
        model="openai/gpt-oss-120b",
    )
    
    return response.choices[0].message.content

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting RAG System...\n")
    
    # Step 1: Read PDF and prepare database
    pdf_chunks = process_pdf("data.pdf")
    collection, embedding_model = setup_database(pdf_chunks)
    
    print("System is ready! 🚀\n")
    
    # Step 2: Ask a question
    user_query = "What is the heading after Development of positive attitude?"
    print(f"User Question: {user_query}")
    
    # Step 3: Get and print the answer
    final_answer = get_answer(user_query, collection, embedding_model)
    print(f"\nAI Answer:\n{final_answer}")