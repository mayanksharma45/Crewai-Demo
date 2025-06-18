from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os

load_dotenv()

topic = "Medical Industry using Generative AI"

# Tool 1
llm = LLM(
    provider="google",
    model="gemini/gemini-1.5-flash-8B",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Tool 2
search_tool = SerperDevTool(n=5)

# # Agent 1
senior_research_analyst = Agent(
    role = "Senior Research Analyst",
    goal = f"Research, analyze and synthesize comprehensive information on {topic} from reliable web sources",
    backstory = "You're an expert research analyst with advanced web research skills. " \
                "You excel at finding, analyzing, and synthesize information from across the internet using search tools. You are skilled at" \
                "disturbing reliable sources from unreliable ones, fact-checking, cross-referencing information, and" \
                "indentifying key patterns and insights. You provide well-organised research briefs with proper citations" \
                "and source verification. Your analysis includes both raw data and interpreted insights, making complex" \
                "information accessable and actionable. ",
    allow_delegation = False,
    verbose = True,
    tools = [search_tool],
    llm = llm
)

# Agnet 2 - Content Writer

content_writer = Agent(
    role = "Content Writer",
    goal = "Transfrom research findings into engaging blog posts while maintaning accuracy",
    backstory = "You are skilled content writer specialsed in creating" \
    "engaging, accessible content from technical research. " \
    "You work closely with the Senior Research Analyst and excel at maintaining the perfect" \
    "balance between informative and entertaining writing, while ensuring all facts and citations from the research" \
    "are properly incorporated. You have a talent for making complex topics approachable without oversimplyfing them. ",
    allow_delegation = False,
    verbose = True,
    llm = llm
)

# Research Tasks

research_tasks = Task(
    description = ("""
        1. Conduct comprehensive research on {topic} includigs:
            - Recent developments and news
            - Key industry trends and innovations
            - Expert opinions and analyses
            - statistical data and market insights
        2. Evaluate source credibilty and fact-check all information
        3. Organise findings into a structured research brief
        4. Include all relevant citations and sources
    """),
    expected_output = """A detailed research report containing:
            - Executing summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - Clear categorization of main themes and patterns
            Please format clear sections and bullet points for easy reference.""",
    agent = senior_research_analyst
)

# Task 2 - Content Writing

writing_task = Task(
    description=("""
        Using the research brief provided, create an engaging blog post that:
            1. Transform technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Inclues:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in {Source: URL} format
            5. Includes a Refrences sections at the end
    """),
    expected_output = """A polished blog post in markdown format that:
        - Engages readers while maintaining accuracy
        - Contains properly structured sections
        - Includes inline citations hyperlinked to the original source url
        - Presents information in an accessible yet informative way
        - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
    agent = content_writer
)

crew = Crew(
    agents = [senior_research_analyst, content_writer],
    tasks = [research_tasks, writing_task],
    verbose = True
)

result = crew.kickoff(inputs = {"topic": topic})

print(result)
