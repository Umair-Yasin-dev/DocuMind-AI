import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="DocuMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

chunks = []
llm_client = None

@app.on_event("startup")
def startup_event():
    global chunks, llm_client
    print("Startup: PDF process ho rahi hai...")
    try:
        reader = PdfReader("data.pdf")
        full_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
        
        # Chunk size 1000 characters
        chunk_size = 1000 
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        # Hum sirf pehle 50 chunks AI ko denge (taqreeban 50,000 characters)
        chunks = chunks[:50] 
        
        llm_client = Groq()
        print(f"System tayyar hai! {len(chunks)} chunks load ho chuke hain. 🚀")
    except Exception as e:
        print(f"Startup Error: {e}")

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not chunks or not llm_client:
        raise HTTPException(status_code=500, detail="System abhi tayyar nahi hai.")
    
    try:
        # Saare 50 chunks ek sath AI ko de rahe hain
                # Saara text nahi, sirf pehle 15 chunks bhejo taake Groq crash na ho
        context = "\n\n".join(chunks[:15])
        
        system_prompt = (
            "You are a helpful assistant. Read the provided context carefully and answer the user's question. "
            "If the answer is truly not in the context, say 'I don't know'. Respond in English."
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