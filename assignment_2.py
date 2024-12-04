from crewai import Agent, Crew, Process, Task, LLM

# Agent Definitions
friendly_interviewer = Agent(
    role="Friendly Interviewer",
    goal="Generate questions to collect structured user information such as name, hobbies, profession, and preferences in a casual and engaging manner.",
    backstory="""You are a virtual assistant trained to ask insightful questions and learn about people. You enjoy making conversations feel natural and engaging, like chatting with a curious friend.""",
    llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.5,
        verbose=True,
    ),
    allow_delegation=False,
)

data_extractor = Agent(
    role="Data Extractor",
    goal="Analyze conversational data and generate a well-structured JSON file containing user details like name, hobbies, profession, and location.",
    backstory="""You are a meticulous data analyst who excels at parsing conversations and turning them into clean, structured data. You take pride in organizing information efficiently and accurately, ensuring nothing is missed.""",
    llm=LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.5,
        verbose=True,
    ),
    allow_delegation=False,
)

# Task Definitions
task_ask_questions = Task(
    name="Ask Questions",
    description="Generate a list of questions to collect user information such as name, hobbies, profession, and preferences.",
    agent=friendly_interviewer,
    expected_output="A list of 10 engaging questions stored as a Python list."
)

task_create_json = Task(
    name="Create JSON",
    description="Process user responses and generate a JSON file containing user details such as name, age, hobbies, profession, and location.",
    agent=data_extractor,
    expected_output="A JSON file with user data (e.g., {'Name': 'XYZ', 'Age': 25})."
)

# Crew Setup
crew = Crew(
    agents=[friendly_interviewer, data_extractor],
    tasks=[task_ask_questions, task_create_json],
    process=Process.sequential,
    verbose=True,
)

# Execute Crew
crew_output = crew.kickoff()
print(f"Raw Output: {crew_output.raw}")
