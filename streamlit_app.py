from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os
import streamlit

load_dotenv()

llm=LLM(
  provider="google",
  model="gemini/gemini-1.5-flash-8B", # u can use other model
  api_key=os.getenv("GROQ_API_KEY")
)
