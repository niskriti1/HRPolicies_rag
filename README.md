# 🧠 HR Policies QA System

A Question-Answering (QA) system built with RAG (Retrieval-Augmented Generation) that helps users query HR policies efficiently and accurately.

## 📌 Features

- RAG pipeline for accurate HR policy answers
- Uses **Parent Document Retriever** for better context handling
- Clean and interactive **Streamlit UI**
- Modular code for easy development and scaling

## 📚 Data

- `data.json`: Contains structured HR policies used for retrieval

--

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **NLP / RAG**:
  - `LangChain`
  - `HuggingFaceEmbeddings`
  - `ChromaDB` as vector store
  - `Gemini` as LLM model

---

## ⚙️ Setup Instructions

# 1. Clone the repo

(https://github.com/niskriti1/HRPolicies_rag.git)

# 2. (Optional) Create a virtual environment

python -m venv venv
source venv/bin/activate

# 3. Install dependencies

pip install -r requirements.txt

# 4. Run the app

streamlit run app.py
