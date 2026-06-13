import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser(description="Run a simple LLM chain.")
parser.add_argument("--country", type=str, required=True, help="Country name to check.")
args = parser.parse_args()

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("Missing API_KEY environment variable.")

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

prompt = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country} in a word?",
)

test_response = PromptTemplate(
    input_variables=["country", "city"],
    template="is {city} capital of {country}? just answer yes or no, no explanation.",
)

# Ask first question and extract the model's text answer.
city_response = (prompt | llm).invoke({"country": args.country})
print(city_response.content)

city = city_response.content.strip()

# Use that city in a second prompt.
response = (test_response | llm).invoke({"country": args.country, "city": city})

print(response.content)
