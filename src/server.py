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
import traceback , logging

evaluator = ScientificHypothesisEvaluator()
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
    
    try:
        # Validate inputs
        if not phenomenon or len(phenomenon.strip()) == 0:
            logger.error("Empty phenomenon provided")
            raise HTTPException(status_code=400, detail="Phenomenon cannot be empty")
        
        if complexity < 0 or complexity > 100:
            logger.error(f"Invalid complexity value: {complexity}")
            raise HTTPException(status_code=400, detail="Complexity must be between 0 and 100")
        
        if not file_media:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="File is required")
        
        # Check file type
        if not file_media.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file_media.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size (limit to 10MB)
        file_size = 0
        pdf_data = await file_media.read()
        file_size = len(pdf_data)
        
        if file_size == 0:
            logger.error("Empty file provided")
            raise HTTPException(status_code=400, detail="File is empty")
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            logger.error(f"File too large: {file_size} bytes")
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        logger.info(f"File validation passed - Size: {file_size} bytes")
        
        # Process the hypothesis generation
        logger.info("Starting hypothesis generation...")
        try:
            hypothesis = generate_hypothesis(phenomenon, complexity, pdf_data)
            logger.info(f"Hypothesis generated successfully: {hypothesis[:100]}...")
        except Exception as e:
            logger.error(f"Error in generate_hypothesis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to generate hypothesis: {str(e)}")
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        try:
            pdf_texts = extract_text_from_pdf_bytes(pdf_data)
            logger.info(f"PDF text extracted successfully - Length: {len(pdf_texts)} characters")
        except Exception as e:
            logger.error(f"Error in extract_text_from_pdf_bytes: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")
        
        # Evaluate hypothesis
        logger.info("Evaluating hypothesis...")
        try:
            info_density = evaluator.evaluate_hypothesis(hypothesis=hypothesis, literature_texts=pdf_texts)
            logger.info("Hypothesis evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in evaluate_hypothesis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to evaluate hypothesis: {str(e)}")
        
        # Calculate complexity score
        logger.info("Calculating complexity score...")
        try:
            complexity_score = calculate_complexity_score(hypothesis)
            logger.info(f"Complexity score calculated: {complexity_score}")
        except Exception as e:
            logger.error(f"Error in calculate_complexity_score: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate complexity score: {str(e)}")
        
        # Convert to dictionary
        try:
            evaluation_dict = asdict(info_density)
            logger.info("Evaluation dictionary created successfully")
        except Exception as e:
            logger.error(f"Error converting info_density to dict: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback response
            evaluation_dict = {"overall_quality": 0, "error": "Failed to process evaluation"}
        
        # Prepare response
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