import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

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

chunks = []
embeddings = []
model = None
llm_client = None

@app.on_event("startup")
def startup_event():
    global chunks, embeddings, model, llm_client
    
    print("Startup: Model load ho raha hai...")
    # Hugging Face ka chota model (Render ke 512MB mein fit ho jata hai)
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    print("Books folder se PDFs parhi ja rahi hain...")
    full_text = ""
    books_folder = "books"
    
    if os.path.exists(books_folder):
        for filename in os.listdir(books_folder):
            if filename.endswith(".pdf"):
                print(f"Parh rahe hain: {filename}")
                reader = PdfReader(os.path.join(books_folder, filename))
                full_text += "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
    else:
        print("Error: 'books' folder nahi mila!")
    
    if full_text.strip():
        # Chunking (1500 characters per chunk)
        chunk_size = 1500 
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        print(f"Total {len(chunks)} chunks banaye gaye.")
        
        print("Embeddings ban rahe hain (thora waqt lagega)...")
        embeddings = model.encode(chunks, show_progress_bar=False)
        
        llm_client = Groq()
        print("System tayyar hai! 🚀")
    else:
        print("Koi text nahi mila books mein.")

@app.get("/")
def read_root():
    return {"status": "healthy", "chunks_loaded": len(chunks)}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not chunks or not llm_client or not model:
        raise HTTPException(status_code=500, detail="System abhi tayyar nahi hai.")
    
    try:
        # 1. Sawal ka embedding banao
        query_embedding = model.encode([request.question])[0]
        
        # 2. Similarity check karo
        scores = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # 3. Top 7 sab se relevant chunks dhoondo
        top_indices = np.argsort(scores)[-7:][::-1]
        context = "\n\n".join([chunks[i] for i in top_indices])
        
        # 4. AI se jawab
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