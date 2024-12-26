## RAG for answer retrieval
This RAG-driven tool is aimed to help students with retrieving answers from PDF documents. It can open your PDFs and answer questions about their content.


## Requirements
Before running the script, make sure you have dependencies listed in `requirements.txt`


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/slava-ugolnikov/RAG-for-answer-retrieval.git
   cd RAG-for-answer-retrieval


## How the code works
- Loads PDF (before, you should put it in 'data/' folder at your local repository)
- Processes it 
- Splits it in chunks
- Converts text chunks to numerical vectors with HuggingFaceEmbeddings
- Gives answer for your query


## Thaks
The code is based on this tutorial 
https://ai.gopubby.com/gpu-less-financial-analysis-rag-model-with-qdrant-langchain-and-gpt4all-x-mistral-7b-a-0b41be10699f
