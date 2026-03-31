from langchain_classic.chains.hyde import prompts
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import RetrievalQA
#load data 
data = PyPDFLoader(r"D:\Anthic\RAG-Work\document\Science-ML-2015.pdf")
docs = data.load()


#split documents

text_split = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
docsSplit = text_split.split_documents(docs)


#create vector store
embedding = HuggingFaceEmbeddings()
vectoreStore = FAISS.from_documents(docsSplit,embedding)


retriever = vectoreStore.as_retriever()
#load model 

model = ChatMistralAI(model="mistral-small-2603", temperature=0.9)

# System + Human prompts
system_prompt = SystemMessagePromptTemplate.from_template("""
You are expert AI assistant. 
Answer questions ONLY using the provided context.
Be concise and accurate.
If the answer is not in the context, say "I don't know"
""")

human_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

#create rag chain
ragSystem = RetrievalQA.from_chain_type(llm = model, retriever = retriever, chain_type_kwargs={"prompt": chat_prompt})

#query
query = "What is main context of the pdf give me 5 line summary"
result = ragSystem.invoke(query)

print(result["result"])