import os
import pdfplumber
import time
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

def load_pdfs_from_folder(folder_path, batch_size=100):
    filenames = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                filenames.append(os.path.join(root, file))

    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i + batch_size]
        documents = []
        for file_path in batch_filenames:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                documents.append(text)
        yield documents, len(filenames)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

class CustomTfidfEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def embed_documents(self, texts):
        embeddings = self.vectorizer.fit_transform(texts)
        return embeddings.toarray()

    def embed_query(self, text):
        embedding = self.vectorizer.transform([text])
        return embedding.toarray()[0]

    def __call__(self, text):
        return self.embed_query(text)

def create_or_update_vectorstore(documents, embedding_function, vectorstore_path="faiss_index.pkl"):
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc))

    if os.path.exists(vectorstore_path):
        with open(vectorstore_path, "rb") as f:
            vectorstore = pickle.load(f)
        new_embeddings = embedding_function.embed_documents(texts)
        vectorstore.add_texts(texts, embeddings=new_embeddings)
    else:
        embeddings = embedding_function.embed_documents(texts)
        vectorstore = FAISS.from_texts(texts, embeddings=embeddings)

    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def initialize_ollama_model():
    llm = Ollama(model="llama3.1")
    return llm

def main():
    pdf_folder_path = "./pdfs"
    vectorstore_path = "faiss_index.pkl"
    print("Initializing Ollama model...")
    llm = initialize_ollama_model()
    embedding_function = CustomTfidfEmbeddings()
    total_files_processed = 0
    print("Loading and processing PDFs in batches...")
    # for documents, total_files in load_pdfs_from_folder(pdf_folder_path):
    #     print(f"Processing batch of {len(documents)} PDF(s)...")
    #     create_or_update_vectorstore(documents, embedding_function, vectorstore_path)
    #     total_files_processed += len(documents)
    # print(f"Total files processed: {total_files_processed} out of {total_files}")
    # print("Loading the final FAISS vector store...")
    with open(vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    inp = input("Enter Question: ")
    start_time = time.time()
    question = inp + " You are an advising bot. You are the play the role of an advisor that helps students at kenensaw state university. Please fact check your answers and mention all required prerequistes for any classes. Please make sure that all course course tags are correct, there are typically no patterns within them." + " Think carefully."
    retrieved_docs = retriever.get_relevant_documents(question)
    print(f"Number of documents retrieved for context: {len(retrieved_docs)}")
    response = qa_chain.run(question)
    print(f"Response: {response}")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()