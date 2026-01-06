from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# print(llm.invoke("Say hello in one sentence"))

response = llm.invoke("Explain AutoML to a bank analyst in one paragraph.")
print(response)
# print(response.content)