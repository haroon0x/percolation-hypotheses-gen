from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from src.config import *
from src.agents.sub_agents import complexity_analyzer_agent
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDEREF2NRjoVnpixzGdpGS0gqiTQVANW5A"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
generation_config = get_generation_config(complexity=3)

hypothesis_generator = LlmAgent(
    name="hypothesis_generator",
    description="Generates scientific hypothesis based on a complexity parameter "
    "Given a scientific phenomenon or topic, your task is to generate hypotheses with varying levels of conceptual complexity, based on the provided complexity scale from 1 (very simple and intuitive) to 10 (highly abstract and cognitively demanding)"
    "To generate accurate hypothesis",
    model="gemini-2.0-flash",
    instruction=Prompt.hypothesis_generator,
    tools=[google_search] 
)


root_agent = hypothesis_generator
