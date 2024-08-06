from fastapi import FastAPI
import uvicorn
import os
import env_file_reader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from utils.docs import qa
from dotenv import load_dotenv

os.environ['OPENAI_API_KEY'] = 'Enter here api key'

app = FastAPI()


@app.get("/{query}")
async def root(query: str):
    #query = "what is c++"
    result = qa({"question": query, "chat_history": []})  # Provide an empty list for chat_history

    response = result.get("answer", "No answer found.")
    print(f"Response: {response}")
    if "source_documents" in result:
        source_documents = result["source_documents"]
        if source_documents:
            source_info = source_documents[0].metadata
            source_file = source_info["source"]
    return{"output": response, "source_file": source_file}


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)