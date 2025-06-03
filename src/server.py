from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.models.model import generate_hypothesis
from src.Hypothesis_Analysis.information_density import ScientificHypothesisEvaluator
from PyPDF2 import PdfReader
import io
from typing import List
from src.Hypothesis_Analysis.complexity_score import calculate_complexity_score
from dataclasses import asdict

evaluator = ScientificHypothesisEvaluator()
app = FastAPI()

class Hypothesis(BaseModel):
    Phenomenon : str
    Complexity : int
    Info_Density : float

@app.post("/generate-hypothesis")
async def get_hypothesis(
    phenomenon: str = Form(...),
    complexity: int = Form(...),
    file_media: UploadFile = File(...) 
    ):
    pdf_data = await file_media.read()
    hypothesis = generate_hypothesis(phenomenon, complexity ,pdf_data)
    pdf_texts = extract_text_from_pdf_bytes(pdf_data)
    info_density = evaluator.evaluate_hypothesis(hypothesis=hypothesis,literature_texts=pdf_texts)
    complexity_score = calculate_complexity_score(hypothesis)
    evaluation_dict = asdict(info_density)
    return JSONResponse(content={"hypothesis": hypothesis,
                                 "Complexity_Score": complexity_score,
                                 "info_density":evaluation_dict
                                 })



@app.get("/")
async def root():
    return {"message": "Hypothesis Generation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    """Extract text from PDF bytes and return as list of page strings"""
    pdf_stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_stream)

    page_texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            page_texts.append(text.strip())
    
    return page_texts

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://percolation-hypotheses.onrender.com"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)