from __future__ import annotations

from typing import Optional, List, Any
from pydantic import BaseModel
from pydantic import Field


class AudioIngestAsyncResp(BaseModel):
    job_id: str
    audio_id: str
    stored_as: str
    visibility: str
    celery_task_id: Optional[str] = None
    status_url: str


class AudioJobResp(BaseModel):
    job_id: str
    audio_id: str
    celery_task_id: Optional[str] = None
    status: str
    progress: int
    message: Optional[str] = None
    cancel_requested: int = 0
    overwrite: int = 0
    delete_old_file: int = 0
    old_stored_path: Optional[str] = None
    cancelled_at: Optional[Any] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None


class AudioDocDetail(BaseModel):
    audio_id: str
    original_filename: str
    stored_path: str
    duration_ms: int
    language: Optional[str] = None
    visibility: str
    status: str
    segment_count: int
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None


class AudioSearchHit(BaseModel):
    audio_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    score: Optional[float] = None
    clip_url: Optional[str] = None


class AudioSearchResp(BaseModel):
    q: str
    k: int
    allowed_visibilities: List[str]
    hits: List[AudioSearchHit]

class AudioAskReq(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=20)
    audio_id: Optional[str] = None  # 只问某条音频；不传就等于全库
    system_prompt: Optional[str] = None  # 可选部分，用来覆盖默认system prompt


class AudioCitation(BaseModel):
    audio_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str
    clip_url: str
    score: Optional[float] = None


class AudioAskResp(BaseModel):
    question: str
    answer: str
    citations: List[AudioCitation]