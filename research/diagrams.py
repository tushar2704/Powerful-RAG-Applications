##© 2024 Tushar Aggarwal. All rights reserved.(https://tushar-aggarwal.com)
##Powerful RAG Applications [https://github.com/tushar2704/Powerful-RAG-Applications] (https://github.com/tushar2704/Powerful-RAG-Applications)
#######################################################################################################
#Importing dependecies
#######################################################################################################




#Simple RAG diagram
from diagrams import Cluster, Diagram
from diagrams.custom import Custom
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.network import Nginx
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
def rag_diagram():
    # Defining custom icons
    DocumentsIcon = Custom("Documents", "./icons/documents.png")
    TextChunksIcon = Custom("Text Chunks", "./icons/text_chunks.png")
    EmbedderIcon = Custom("Embedder", "./icons/embedder.png")
    RerankerIcon = Custom("Reranker", "./icons/reranker.png")
    OptionalRerankerIcon = Custom("Optional Reranker", "./icons/optional_reranker.png")
    LLMIcon = Custom("LLM", "./icons/llm.png")

    with Diagram("Text-based Question Answering System", show=False, outformat="png"):
        docs = DocumentsIcon()
        text_chunks = TextChunksIcon()
        embedder = EmbedderIcon()
        embeddings = PostgreSQL("Embeddings")
        reranker = RerankerIcon()
        optional_reranker = OptionalRerankerIcon()
        llm = LLMIcon()
        prompt = Custom("Prompt Template", "./icons/prompt.png")

        with Cluster("Main Flow"):
            docs >> text_chunks >> embedder >> embeddings
            embeddings >> reranker
            reranker >> optional_reranker >> llm
            llm >> Custom("Response", "./icons/response.png")

        with Cluster("Prompt"):
            optional_reranker >> prompt >> llm


















