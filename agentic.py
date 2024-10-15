import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.llms import OpenAI
'''
We use https://serpapi.com/ for searching (as a tool).
'''


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')

# Create an OpenAI api
llm = OpenAI(temperature=0)
# Set up which tools shall be used by the agent
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
# initialize the agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# run a query which uses todays data
answer = agent.run("What today's the value of IE00BP3QZB59 ETF? Also give me the date of the value! What is this value divided by 3?")
print(answer)