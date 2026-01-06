# import dotenv
# from langchain_openai import ChatOpenAI

# dotenv.load_dotenv()

# chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

from email import message
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_openai_functions_agent, 
    Tool, 
    AgentExecutor,
)
from langchain import hub 
from langchain_intro.tools import get_current_wait_time

dotenv.load_dotenv()

review_chroma_path = "C:\\Users\\noeln\\Desktop\\llm\\20260106_llm_chatbot\\langchain_intro\\chromadb\\chroma_data"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reviews_vector_db = Chroma(
    persist_directory = review_chroma_path,
    embedding_function = embeddings
)

review_retriever = reviews_vector_db.as_retriever(k=10)

review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables = ["context"],
        template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables = ["question"],
        template="{question}"
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables = ['context', 'question'],
    messages = messages
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# to format the model output as a string, without AI message etc.
output_parser = StrOutputParser()

# review_chain = review_prompt_template | chat_model | output_parser

review_chain = (
    {"context": review_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | output_parser
)

# response = chat_model.invoke("Explain AutoML to a bank analyst in one paragraph.")
# print(response.content)