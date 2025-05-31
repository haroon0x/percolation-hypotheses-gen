from google.adk import Agent
from google.adk.tools import google_search
from src.config import *



complexity_analyzer_agent = Agent(
    model="gemini-2.5-flash-preview-05-20",
    name="complexity_analyzer_agent",
    instruction=Prompt.complexity_analyzer,
    output_key="complexity_value"
)
literature_fetcher_agent = Agent(
    model="gemini-2.5-flash-preview-05-20",
    name="literature_fetcher_agent",
    instruction=Prompt.literature_fetcher,
    tools=[google_search],
    
    output_key="recent_citing_papers"
)
