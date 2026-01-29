from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # 将操作系统中的环境变量读取一下

class Settings(BaseModel):  # BaseModel后续这个类会转换为JSON/字典方便使用
    qianwen_api_key: str = os.getenv("QIANWEN_API_KEY", "sk-6ceac0ac789d4a4ead54060a4af23724")
    qianwen_embedding_model_name: str = os.getenv("QIANWEN_EMBEDDING_MODEL_NAME", 'text-embedding-v2')
    model_name: str = os.getenv("MODEL_NAME", "qwen-plus")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    audio_collection_name: str = os.getenv("AUDIO_COLLECTION_NAME", "audio_base")

    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", "amqp://LoveBreaker:123456@127.0.0.1:5672/%2F")  # ⚠️改自己的用户名和密码
    celery_audio_queue: str = "audio"  # 消息队列的名字

    audio_dir: str = "data/audio"
    audio_wav_dir: str = "data/audio_wav"
    audio_clip_dir: str = "data_audiomp3/audio_clips"


settings = Settings()
