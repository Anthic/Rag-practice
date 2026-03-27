from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -------------------------------
# Load document
loader = TextLoader(r"D:\Anthic\RAG-Work\document\data.txt", encoding="utf-8")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector store
embedding = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

# Load model
model = ChatMistralAI(model="mistral-small-2603", temperature=0.8)

# System + Human prompts
system_prompt = SystemMessagePromptTemplate.from_template("""
You are expert AI assistant. 
Answer questions ONLY using the provided context.
Be concise and accurate.
If the answer is not in the context, say "I don't know"
""")

human_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt}
)

# Query
query = "What is RAG?"
result = qa.invoke(query)

print(result["result"])