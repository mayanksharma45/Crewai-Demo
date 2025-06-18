from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(
    provider="google",
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

print(llm.call("What is CrewAI and how does it work?"))
