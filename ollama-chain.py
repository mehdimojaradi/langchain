import argparse

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

parser = argparse.ArgumentParser(description="Run a simple Ollama LLM chain.")
parser.add_argument("--country", type=str, required=True, help="Country name to check.")
args = parser.parse_args()

llm = ChatOllama(model="llama3.2:1b")

prompt = PromptTemplate(
	input_variables=["country"],
	template="What is the capital of {country}? Before extracting the answer, verify the information and make sure your answer is correct. then just give the answer in a word., no explanation.",
)

test_response = PromptTemplate(
	input_variables=["country", "city"],
	template="is {city} capital of {country}? Before answering, verify the information and make sure your answer is correct based on reliable sources. then just answer yes or no",
)

# Ask first question and extract the model's text answer.
city_response = (prompt | llm).invoke({"country": args.country})
print(city_response.content)

city = city_response.content.strip()

# Use that city in a second prompt.
response = (test_response | llm).invoke({"country": args.country, "city": city})

print(response.content)