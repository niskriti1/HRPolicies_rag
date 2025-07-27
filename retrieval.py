import json
from typing import List,Tuple
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

def load_data() -> List[dict]:
  with open('data.json' , 'r') as file:
    return json.load(file)

  
def clean_text(text:str) -> str:
  return text.lower().strip()

def process_document(raw_data:List[dict]) -> List[Document]:
  documents=[]
  for data in raw_data:
    for faq in data["faq"]:  # now looping over list of FAQs
            q = clean_text(faq["question"])
            a = clean_text(faq["answer"])
            content = f"Question: {q} \n Answer: {a}"
            documents.append(Document(page_content=content, metadata={'category': data['category']}))

  return documents 
  
def initialize_retriver(google_api_key: str) -> ParentDocumentRetriever:
  raw_data = load_data()
  
  documents = process_document(raw_data)
  
  # load the embedding model
  embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  

  # load chat model
  llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key
  )
  
  # Create the vector store
  vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="chroma_store"
    )
  
  # Splitters
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
  
  # Local byte store for parent-child mapping
  store = LocalFileStore("chroma_store/byte_store")
  docstore = create_kv_docstore(store)
  
  retriever = ParentDocumentRetriever(
    llm=llm,
    documents=documents,
    embedding=embedding,
    vectorstore=vectorstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    docstore=docstore
  )
  
  retriever.add_documents(documents)
  
  return retriever
  
  
  
  
def get_context_data(retriever : ParentDocumentRetriever, question:str) -> Tuple[str,bool]:
  docs=retriever.get_relevant_documents(question)
  if not docs:
    return "",False
  
  context = "\n\n".join(doc.page_content for doc in docs) 
  return context,True