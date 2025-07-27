import streamlit as st
from datetime import datetime
from typing import Tuple
import os
from dotenv import load_dotenv
from retrieval import initialize_retriver,get_context_data
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
  st.error("Missing api key. Check your .env file")
  st.stop()
  
try:
  retriever = initialize_retriver(google_api_key)
except FileNotFoundError as e:
  st.error(f"Configuration file not found: {str(e)}")
  st.stop()
except Exception as e:
    st.error(f"Error initializing retriever: {str(e)}")
    st.stop()
  
def rag_pipeline(user_question:str) -> str:
  llm = ChatGoogleGenerativeAI(
      model="gemini-1.5-flash",
      google_api_key=google_api_key,
      temperature=0.2
  )
  
  
  
  
  context,found=get_context_data(retriever,user_question)

  prompt=ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
      "Answer the question using ONLY the context below. If unsure, say \"I don't know.\"\n\nContext:\n{context}"
    ),
    HumanMessagePromptTemplate.from_template(
      "Question: {input}")
  ])

  
  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  chain = create_retrieval_chain(retriever, question_answer_chain)  
  
  response=chain.invoke({"input": user_question})
  
  return response['answer']

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
    


# Custom CSS for chat bubbles
st.markdown("""
<style>
.user-message {
    background-color: #778899;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
}
.assistant-message {
    background-color: #404756;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
}
.message-time {
    font-size: 0.8em;
    color: #00000;
}
</style>
""", unsafe_allow_html=True)



# Title and description
st.title("HR FAQ Chatbot")
st.markdown("Ask any HR-related question (leave,working benefits,remote work)")

# New Chat button
if st.button("🔄 New Chat"):
    st.session_state.messages = []
    st.session_state.current_question = None
    st.rerun()
    
# Display chat history
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
             👩‍💼 You: {message["content"]}
                <div class="message-time">{message["time"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                🤖 Bot: {message["content"]}
                <div class="message-time">{message["time"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask your question here...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": datetime.now().strftime("%H:%M")
    })
    st.session_state.current_question = user_input
    
    # Show spinner while generating answer
    with st.spinner("Please wait... Generating answer..."):
        # Get answer from local knowledge base
        answer = rag_pipeline(user_input)
    
    # Add assistant message to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%H:%M")
    })
    
st.markdown("---")
st.caption("Built with LangChain, Chroma, HuggingFace, Gemini 1.5.")