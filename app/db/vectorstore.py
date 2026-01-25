import chromadb
from langchain_chroma import Chroma
from app.config import settings

def get_client():
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port
    )

def get_vectorstore(embeddings):
    # Connect to Chroma Server running in Docker (chromadb/chroma:0.6.3)
    return Chroma(  # 返回的就是一个指向chromadb的连接
        client=get_client(),
        collection_name=settings.collection_name,  # 为我们这个项目建议一个专属的数据库
        embedding_function=embeddings,
    )

def get_audio_vectorstore(embeddings):  # ⚠️实际上只需要在这里添加即可
    return Chroma(
        client=get_client(),
        collection_name=settings.audio_collection_name,
        embedding_function=embeddings,
    )