import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

reviews_csv_path = "C:\\Users\\noeln\\Desktop\\llm\\20260106_llm_chatbot\\data\\reviews.csv"
reviews_chroma_path = "C:\\Users\\noeln\\Desktop\\llm\\20260106_llm_chatbot\\langchain_intro\\chromadb\\chroma_data"

dotenv.load_dotenv()

loader = CSVLoader(file_path = reviews_csv_path, source_column = "review")
reviews = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reviews_vector_db = Chroma.from_documents(
    reviews, embeddings, persist_directory=reviews_chroma_path
)