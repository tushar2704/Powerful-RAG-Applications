##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##Powerful RAG Applications [https://github.com/tushar2704/Powerful-RAG-Applications] (https://github.com/tushar2704/Powerful-RAG-Applications)
#######################################################################################################
#Importing dependecies
#######################################################################################################




#Simple RAG diagram
from diagrams import Cluster, Diagram
from diagrams.custom import Custom
from diagrams.onprem.database import Mongodb
from diagrams.programming.language import Python
def rag_diagram():
    with Diagram("Text-based Information Retrieval System", show=False):
        with Cluster("Input"):
            documents = Custom("Documents", "./icons/documents.png")
            text_chunks = Custom("Text Chunks", "./icons/text_chunks.png")

        embedder = Python("Embedder")
        
        with Cluster("Storage"):
            embeddings = Mongodb("Embeddings\nin Memory\nor Vector DB")

        reranker = Python("Reranker")
        prompt = Custom("Plug into\nPrompt Template", "./icons/prompt.png")
        llm = Python("LLM")
        response = Custom("Response", "./icons/response.png")

        documents >> text_chunks >> embedder >> embeddings
        embeddings >> reranker >> prompt >> llm >> response


















