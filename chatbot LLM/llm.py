from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
import os
from typing_extensions import Concatenate
import constants 

# # Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = constants.API_KEY

# # Provide the path to the directory containing the PDF files
# pdf_directory = 'data/'
api_key = os.environ["OPENAI_API_KEY"] = constants.API_KEY

def document_search(api_key,pdf_directory,QUERY):
    # List all PDF files in the directory
    pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

    # Read text from all PDF files
    raw_text = ''
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        pdfreader = PdfReader(pdf_path)
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query =  QUERY
    
    if query.lower() == 'exit':
        print("Exiting the program.")
    
    docs = document_search.similarity_search(query)
    
    return chain.run(input_documents=docs, question=query)