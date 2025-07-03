from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
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

from agents import Agent, handoff

# Step 1: Do agents banayein
refund_agent = Agent(name="Refund agent")
billing_agent = Agent(name="Billing agent")

# Step 2: Main agent banayein jo handoff karega
customer_agent = Agent(
    name="Customer Support",
    handoffs=[
        handoff(
            agent=refund_agent,
            tool_name_override="custom_refund_tool",
            tool_description_override="Handles refund-related queries"
        ),
        billing_agent
    ]
)
user_input = input('Enter your prompt: ')
result = Runner.run_sync(customer_agent, user_input, run_config=config)
print('last agent: ',result.last_agent)
print('final output: ',result.final_output)