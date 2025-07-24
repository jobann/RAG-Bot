# RAG-Bot

RAG-Bot is a simple question-answering system that uses Retrieval-Augmented Generation (RAG) combined with the LLaMA language model. It allows you to provide a paragraph and then ask questions, generating natural, context-aware answers instead of just extracting text.

---

## Features

- Input any paragraph and ask questions about it
- Retrieves relevant context using vector embeddings (FAISS)
- Generates natural language answers with LLaMA
- Easy to set up and extend

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/jobann/RAG-Bot.git
cd RAG-Bot

2.	Create and activate a Python virtual environment:

python3 -m venv myenv
source myenv/bin/activate  # For Windows use: myenv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt