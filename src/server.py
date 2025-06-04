from fastapi import FastAPI, Form, UploadFile, File , HTTPException
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
import traceback , logging ,sys

app = FastAPI()

class Hypothesis(BaseModel):
    Phenomenon : str
    Complexity : int
    Info_Density : float

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


@app.post("/generate-hypothesis")
async def get_hypothesis(
    phenomenon: str = Form(...),
    complexity: int = Form(...),
    file_media: UploadFile = File(...) 
    ):
    logger.info(f"Received request - Phenomenon: {phenomenon}, Complexity: {complexity}, File: {file_media.filename}")
    
    evaluator = ScientificHypothesisEvaluator()
    try:
        if not phenomenon or len(phenomenon.strip()) == 0:
            raise HTTPException(status_code=400, detail="Phenomenon cannot be empty")
        
        if complexity < 0 or complexity > 100:
            raise HTTPException(status_code=400, detail="Complexity must be between 0 and 100")
        
        if not file_media:
            raise HTTPException(status_code=400, detail="File is required")
        
  
        if not file_media.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_size = 0
        pdf_data = await file_media.read()
        file_size = len(pdf_data)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        try:
            hypothesis = generate_hypothesis(phenomenon, complexity, pdf_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate hypothesis: {str(e)}")
        
        try:
            pdf_texts = extract_text_from_pdf_bytes(pdf_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")

        try:
            info_density = evaluator.evaluate_hypothesis(hypothesis, pdf_texts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to evaluate hypothesis: {str(e)}")
    
       
        try:
            complexity_score = calculate_complexity_score(hypothesis)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to calculate complexity score: {str(e)}")
        
        
        try:
            evaluation_dict = asdict(info_density)
        except Exception as e:
            evaluation_dict = {"overall_quality": 0, "error": "Failed to process evaluation"}
        
        response_data = {
            "hypothesis": hypothesis,
            "Complexity_Score": complexity_score,
            "info_density": evaluation_dict,
            "status": "success",
            "file_processed": file_media.filename,
        }
        
        logger.info("Request processed successfully")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_hypothesis: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
            "url": str(request.url),
            "method": request.method
        }
    )


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
     allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)