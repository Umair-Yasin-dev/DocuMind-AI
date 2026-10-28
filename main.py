import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastembed_cloud import CloudTextEmbedding
import numpy as np

load_dotenv()

app = FastAPI(title="DocuMind AI - Multi-Book API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

# Global variables
chunks = []
embeddings = []
model = None
llm_client = None
is_loaded = False

def load_books():
    """Yeh function books ko background mein load karega."""
    global chunks, embeddings, model, llm_client, is_loaded
    
    print("Background: Books load ho rahi hain...")
    try:
        full_text = ""
        books_folder = "books"
        
        if os.path.exists(books_folder):
            for filename in os.listdir(books_folder):
                if filename.endswith(".pdf"):
                    print(f"Parh rahe hain: {filename}")
                    reader = PdfReader(os.path.join(books_folder, filename))
                    full_text += "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
        
        if full_text.strip():
            chunk_size = 1500 
            chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
            print(f"Total {len(chunks)} chunks banaye gaye.")
            
            model = CloudTextEmbedding()
            embeddings = model.embed(chunks)
            llm_client = Groq()
            is_loaded = True
            print("Background: System tayyar hai! 🚀")
        else:
            print("Background: Koi text nahi mila.")
    except Exception as e:
        print(f"Background Error: {e}")

@app.on_event("startup")
def startup_event():
    # Server foran start ho jaye, books background mein load hon
    import threading
    thread = threading.Thread(target=load_books)
    thread.start()
    print("Server start ho gaya. Books background mein load ho rahi hain...")

@app.get("/")
def read_root():
    return {"status": "healthy", "loaded": is_loaded, "chunks": len(chunks)}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not is_loaded or not llm_client or not model:
        raise HTTPException(status_code=503, detail="System abhi load ho raha hai. Thora intezar karein.")
    
    try:
        query_embedding = model.query_embed(request.question)
        
        scores = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        top_indices = np.argsort(scores)[-7:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])
        
        system_prompt = (
            "You are a helpful assistant. Answer the question based ONLY on the provided context. "
            "If the answer is not in the context, say 'I don't know'. Respond in English."
        )
        
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ],
            model="openai/gpt-oss-120b"
        )
        
        return {"answer": response.choices[0].message.content}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))