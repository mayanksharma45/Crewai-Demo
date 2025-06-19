from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os

load_dotenv()

llm=LLM(model="")
