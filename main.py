import os
import re
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
full_text_global = ""

@app.on_event("startup")
def startup_event():
    global chunks, llm_client, full_text_global
    print("Startup: PDF process ho rahi hai...")
    try:
        reader = PdfReader("data.pdf")
        full_text_global = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
        
        chunk_size = 1000 
        chunks = [full_text_global[i:i+chunk_size] for i in range(0, len(full_text_global), chunk_size)]
        
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
        q = request.question.lower()
        
        # Smart Search: Agar sawal Importance ke baare mein hai
        if "importance" in q or "advantage" in q:
            match = re.search(r"1\.3\s+IMPORTANCE OF COMMUNICATION", full_text_global, re.IGNORECASE)
            if match:
                start = match.start()
                context = full_text_global[start:start+2500] # Us heading ke baad ka text
            else:
                context = "\n\n".join(chunks[:30])
                
        # Agar sawal Purpose ke baare mein hai
        elif "purpose" in q:
            match = re.search(r"1\.4\s+PURPOSE OF COMMUNICATION", full_text_global, re.IGNORECASE)
            if match:
                start = match.start()
                context = full_text_global[start:start+2500]
            else:
                context = "\n\n".join(chunks[:30])
        
        # Default: Agar koi specific heading nahi mili, toh pehle 20 chunks de do
        else:
            context = "\n\n".join(chunks[:20])

        system_prompt = (
            "You are a helpful assistant. Read the provided context carefully and answer the user's question. "
            "Respond in clear English."
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