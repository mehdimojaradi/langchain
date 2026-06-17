from langchain_classic.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import LLMChain

llm = ChatOllama(model="llama3.2:1b")

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=llm,
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        # this line is important to keep the conversation history in the prompt.
        # MesssagesPlaceholer is a special prompt template that will be replaced with the conversation history.
        # meaning that the model will see the previous messages in the conversation and can use them to generate a response.
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True,
)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])
