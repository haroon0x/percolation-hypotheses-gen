from google import genai
from google.genai import types
from src.config import *
import dotenv
import os


dotenv.load_dotenv()
GEMINI_API_KEY= os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
client_ = genai.Client(api_key=GEMINI_API_KEY)