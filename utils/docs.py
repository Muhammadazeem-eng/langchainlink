import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
os.environ['OPENAI_API_KEY'] = 'enter your api'


base_path = os.getenv(r'BASE_PATH')


# Define a base path for local files (change this to your web server base URL when hosting)
#base_path = r"C:\Users\muham\Downloads\Content" # Local base path for files
# Providing paths
pdf_loader = DirectoryLoader(base_path, glob="**/*.pdf")
readme_loader = DirectoryLoader(base_path, glob="**/*.docx")
txt_loader = DirectoryLoader(base_path, glob="**/*.txt")
# Take all the loaders
loaders = [pdf_loader, readme_loader, txt_loader]
# Create a simple Document class
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

documents = []
for loader in loaders:
    docs = loader.load()
    for doc in docs:
        # Assuming each document has 'page_content' and 'metadata'
        for page in doc.page_content.split("\f"):  # '\f' is the page break character
            documents.append(Document(
                page_content=page,
                metadata={"source": doc.metadata["source"]}
            ))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
split_documents = []
for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        split_documents.append(Document(
            page_content=chunk,
            metadata=doc.metadata
        ))

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings
)
# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

# # Create QA chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, return_source_documents=True)
