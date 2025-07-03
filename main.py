from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from dotenv import load_dotenv
load_dotenv()
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Check if the API key is present; if not, raise an error
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client   
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

english_tutor = Agent(
    name = 'English Tutor Assistant',
    instructions='You answer all english related questions..'
)

physics_tutor = Agent(
    name = 'Physics Tutor Assistant',
    instructions='You answered all physics related questions'
)

chemistry_tutor = Agent(
    name = 'chemistry Tutor Assistant',
    instructions='You answered all chemistry related questions'
)

triage = Agent(
    name = 'Triage Assistant',
    instructions='You determine which agent should handle the user request based on the nature of the inquiry.',
    handoffs=[english_tutor, physics_tutor, chemistry_tutor]
)

user_prompt = input('Enter your prompt: ')
result = Runner.run_sync(triage, user_prompt, run_config=config)
print('last agent: ',result.last_agent)
print('final output: ',result.final_output)