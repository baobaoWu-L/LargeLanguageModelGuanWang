from __future__ import annotations

import time, uuid
from pathlib import Path
from typing import List, Optional
import httpx
from fastapi import HTTPException
from dashscope import Generation
from fastapi import (APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile,)
from fastapi.responses import FileResponse

from app.api.auth import UserInDB, get_current_user
from app.audio.clip import clip_audio_to_mp3
from app.config import settings
from app.db import audio_db, audio_job_db
from app.deps import get_audio_vs
from app.model.audio_model import AudioIngestAsyncResp, AudioJobResp, AudioDocDetail, AudioSearchResp, AudioSearchHit, \
    AudioCitation, AudioAskResp, AudioAskReq
from app.rbac.perm import allowed_kb_visibilities, check_permission
from app.audio.audio_tasks import audio_ingest_task


router = APIRouter(prefix="/audio", tags=["audio"])

AUDIO_DIR = Path(getattr(settings, "audio_dir", "data/audio"))
CLIP_DIR = Path(getattr(settings, "audio_clip_dir", "data/audio_clips"))

def _clip_url(base: str, audio_id: str, start_ms: int, end_ms: int) -> str:
    return f"{base}/audio/docs/{audio_id}/clip?start_ms={start_ms}&end_ms={end_ms}"


def _openai_chat_complete(*, model: str, api_key: str, messages: list[dict[str, str]], timeout_s: float = 60.0) -> str:


    # url = "https://api.openai.com/v1/chat/completions"
    # headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # payload = {
    #     "model": model,
    #     "messages": messages,
    #     "temperature": 0.2,
    # }
    # with httpx.Client(timeout=timeout_s) as client:
    #     r = client.post(url, headers=headers, json=payload)
    #     if r.status_code >= 400:
    #         raise HTTPException(status_code=500, detail=f"OpenAI error: {r.status_code} {r.text[:300]}")
    #     data = r.json()
    # try:
    #     return (data["choices"][0]["message"]["content"] or "").strip()
    # except Exception:
    #     raise HTTPException(status_code=500, detail="OpenAI response parse error")

    # llm = get_llm()  # 获取通义千问的LLM实例
    s = settings
    # 构建调用参数（与 Generation.call 参数对齐）
    payload = {
        "model": s.model_name,
        "messages": messages,
        # "temperature": llm.temperature,
    }

    try:
        # 优先使用 LLM 实例自带的 API Key（若存在），否则回退到全局配置
        # api_key_to_use = getattr(llm, 'qianwen_api_key', None) or settings.qianwen_api_key

        # 通义千问调用（使用 dashscope SDK，解包 payload 并补充必要参数）
        response = Generation.call(
            **payload,
            api_key=s.qianwen_api_key,
            timeout=timeout_s
        )

        # 检查响应状态
        if response.status_code != 200:
            error_msg = response.output.get("error", {}).get("message", "Unknown error")
            raise HTTPException(
                status_code=500,
                detail=f"Qwen error: {response.status_code} {error_msg[:300]}"
            )

        # 提取生成内容
        content = response.output.get("text", "").strip()
        return content

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Qwen API error: {str(e)}"
        )


def _build_rag_messages(question: str, citations: list[AudioCitation], system_prompt: Optional[str]) -> list[dict[str, str]]:
    sys = (system_prompt or "").strip() or (
        "你是企业知识库助手，回答必须基于给定的【音频片段】内容。"
        "如果片段不足以回答，就明确说“不确定/片段中没有”。"
        "return中的'text'所有文字请使用中文简体回答"
        "回答要简洁，并在结尾给出引用列表（用 [1][2]... 标注）。"
    )

    ctx_lines: list[str] = []
    for i, c in enumerate(citations, start=1):
        ctx_lines.append(
            f"[{i}] audio_id={c.audio_id} segment_id={c.segment_id} "
            f"start_ms={c.start_ms} end_ms={c.end_ms}\n"
            f"片段文本：{c.text}"
        )
    ctx = "\n\n".join(ctx_lines) if ctx_lines else "（无片段）"

    user = (
        f"问题：{question}\n\n"
        f"【音频片段】\n{ctx}\n\n"
        "要求：\n"
        "1) 只用片段信息回答。\n"
        "2) 如果引用了某个片段，请用 [序号] 标注。\n"
        "3) 不要编造片段里没有的信息。"
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _require_manage_docs(user: UserInDB) -> None:
    check_permission(user, "kb.manage_docs")


def _normalize_visibility(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("public", "internal"):
        return v
    return "public"


def _compute_allowed_visibilities(user: UserInDB) -> List[str]:
    perms = getattr(user, "permissions", None)
    allowed = allowed_kb_visibilities(perms)
    if "public" not in allowed:
        allowed = ["public"] + [x for x in allowed if x != "public"]
    return allowed


def _ensure_can_access_visibility(user: UserInDB, doc_visibility: str) -> List[str]:
    allowed = _compute_allowed_visibilities(user)
    vis = (doc_visibility or "").strip().lower()
    if vis not in set(allowed):
        raise HTTPException(status_code=403, detail="no permission to access this audio")
    return allowed


def _absolute_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")



@router.post("/ingest", response_model=AudioIngestAsyncResp)
async def ingest_audio(
    file: UploadFile = File(...),
    visibility: str = Form("public"),
    audio_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    overwrite: bool = Form(False),
    delete_old_file: bool = Form(False),
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    visibility = _normalize_visibility(visibility or "public")
    audio_id = (audio_id or f"aud-{uuid.uuid4().hex[:12]}").strip()
    job_id = f"job-{uuid.uuid4().hex[:12]}"

    if audio_db.is_audio_running(audio_id):
        raise HTTPException(status_code=409, detail="audio is running, try later")

    existed = audio_db.get_audio_document(audio_id)
    if existed and not overwrite:
        raise HTTPException(status_code=409, detail="audio_id already exists; set overwrite=true")

    old_stored_path = existed["stored_path"] if existed else None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix or ".bin"
    raw_path = AUDIO_DIR / f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    raw_path.write_bytes(raw_bytes)

    audio_db.upsert_audio_document(
        audio_id=audio_id,
        original_filename=file.filename,
        stored_path=str(raw_path),
        duration_ms=0,
        language=language,
        visibility=visibility,
        status="queued",
        uploader_user_id=int(getattr(current_user, "id", 0) or 0) or None,
        uploader_username=getattr(current_user, "username", None),
        segment_count=0,
    )

    audio_job_db.create_job(
        job_id,
        audio_id,
        overwrite=bool(overwrite),
        delete_old_file=bool(delete_old_file),
        old_stored_path=old_stored_path if overwrite else None,
    )

    async_result = audio_ingest_task.apply_async(
        args=[job_id, audio_id],
        queue=getattr(settings, "celery_audio_queue", "audio"),
    )
    audio_job_db.bind_task(job_id, async_result.id)

    return AudioIngestAsyncResp(
        job_id=job_id,
        audio_id=audio_id,
        stored_as=str(raw_path),
        visibility=visibility,
        celery_task_id=async_result.id,
        status_url=f"/audio/jobs/{job_id}",
    )


@router.get("/jobs/{job_id}", response_model=AudioJobResp)
def get_audio_job(
    job_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    row = audio_job_db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return row


@router.post("/jobs/{job_id}/cancel")
def cancel_audio_job(
    job_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    ok = audio_job_db.request_cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "cancel_requested": True}


@router.get("/docs/{audio_id}", response_model=AudioDocDetail)
def get_audio_doc(
    audio_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    _require_manage_docs(current_user)
    row = audio_db.get_audio_document(audio_id)
    if not row:
        raise HTTPException(status_code=404, detail="audio not found")
    return row


@router.get("/query", response_model=AudioSearchResp)
def query_audio(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(6, ge=1, le=20),
    current_user: UserInDB = Depends(get_current_user),
) -> AudioSearchResp:
    allowed_vis = _compute_allowed_visibilities(current_user)
    allowed_vis_set = set(allowed_vis)

    vs = get_audio_vs()
    base = _absolute_base(request)

    fetch_k = min(max(k * 5, k), 50)
    where = {"visibility": {"$in": allowed_vis}}

    try:
        docs_scores = vs.similarity_search_with_score(q, k=fetch_k, filter=where)
    except TypeError:
        docs_scores = vs.similarity_search_with_score(q, k=fetch_k, where=where)

    hits: list[AudioSearchHit] = []
    seen: set[tuple[str, str, int, int]] = set()

    for doc, score in docs_scores:
        md = doc.metadata or {}
        audio_id = str(md.get("audio_id") or "").strip()
        segment_id = str(md.get("segment_id") or "").strip()
        if not audio_id or not segment_id:
            continue

        try:
            start_ms = int(md.get("start_ms") or 0)
            end_ms = int(md.get("end_ms") or 0)
        except Exception:
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue

        key = (audio_id, segment_id, start_ms, end_ms)
        if key in seen:
            continue
        seen.add(key)

        db_doc = audio_db.get_audio_document(audio_id)
        if not db_doc:
            continue

        doc_vis = (db_doc.get("visibility") or "").strip().lower()
        if doc_vis not in allowed_vis_set:
            continue

        clip_url = _clip_url(base, audio_id, start_ms, end_ms)
        text = (doc.page_content or "").strip()

        hits.append(
            AudioSearchHit(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                score=float(score) if score is not None else None,
                clip_url=clip_url,
            )
        )
        if len(hits) >= k:
            break

    return AudioSearchResp(q=q, k=k, allowed_visibilities=allowed_vis, hits=hits)


@router.get("/docs/{audio_id}/clip")
def get_audio_clip(
    audio_id: str,
    background_tasks: BackgroundTasks, # 后台任务，和多线程不太一样，可以被当作轻量的多线程
    start_ms: Optional[int] = Query(default=None, ge=0),
    end_ms: Optional[int] = Query(default=None, ge=0),
    segment_id: Optional[str] = Query(default=None),
    current_user: UserInDB = Depends(get_current_user),):
    doc = audio_db.get_audio_document(audio_id)
    if not doc:
        raise HTTPException(status_code=404, detail="audio not found")

    _ensure_can_access_visibility(current_user, doc.get("visibility") or "")

    if segment_id:
        if ":" not in segment_id:
            raise HTTPException(status_code=400, detail="invalid segment_id format")
        seg_audio_id, seg_idx_str = segment_id.split(":", 1)
        if seg_audio_id != audio_id:
            raise HTTPException(status_code=400, detail="segment_id does not match audio_id")
        try:
            seg_idx = int(seg_idx_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid segment_idx in segment_id")

        seg = audio_db.get_audio_segment(audio_id, seg_idx)
        if not seg:
            raise HTTPException(status_code=404, detail="segment not found")

        start_ms = int(seg["start_ms"])
        end_ms = int(seg["end_ms"])
    else:
        if start_ms is None or end_ms is None:
            raise HTTPException(
                status_code=400,
                detail="start_ms and end_ms are required when segment_id is not provided",
            )

    if end_ms <= start_ms:
        raise HTTPException(status_code=400, detail="end_ms must be greater than start_ms")

    max_clip_ms = 5 * 60 * 1000
    if (end_ms - start_ms) > max_clip_ms:
        raise HTTPException(status_code=400, detail="clip too long")

    src_path = Path(doc["stored_path"])
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="stored audio file missing")

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    clip_name = f"{audio_id}_{start_ms}_{end_ms}_{uuid.uuid4().hex[:8]}.mp3"
    clip_path = CLIP_DIR / clip_name

    try:
        clip_audio_to_mp3(
            src_path=src_path,
            dst_path=clip_path,
            start_ms=int(start_ms),
            end_ms=int(end_ms),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="failed to generate clip")

    background_tasks.add_task(lambda p=str(clip_path): Path(p).unlink(missing_ok=True))

    return FileResponse(
        path=str(clip_path),
        media_type="audio/mpeg",
        filename=clip_name,
    )

@router.post("/ask", response_model=AudioAskResp)
def ask_audio(
    req: AudioAskReq,
    request: Request,
    current_user: UserInDB = Depends(get_current_user),) -> AudioAskResp:
    question = (req.question or "").strip()
    k = max(1, min(int(req.k or 6), 20))

    allowed_vis = _compute_allowed_visibilities(current_user)
    allowed_vis_set = set(allowed_vis)
    vs = get_audio_vs()
    base = _absolute_base(request)

    where = {"visibility": {"$in": allowed_vis}}
    if req.audio_id:
        where = {"$and": [
            {"visibility": {"$in": allowed_vis}},
            {"audio_id": req.audio_id},
        ]}

    fetch_k = min(max(k * 5, k), 50)
    try:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, filter=where)
    except TypeError:
        docs_scores = vs.similarity_search_with_score(question, k=fetch_k, where=where)

    citations: list[AudioCitation] = []
    seen: set[tuple[str, str, int, int]] = set()

    for doc, score in docs_scores:
        md = doc.metadata or {}
        audio_id = str(md.get("audio_id") or "").strip()
        segment_id = str(md.get("segment_id") or "").strip()
        if not audio_id or not segment_id:
            continue

        try:
            start_ms = int(md.get("start_ms") or 0)
            end_ms = int(md.get("end_ms") or 0)
        except Exception:
            continue
        if start_ms < 0 or end_ms <= start_ms:
            continue

        key = (audio_id, segment_id, start_ms, end_ms)
        if key in seen:
            continue
        seen.add(key)

        db_doc = audio_db.get_audio_document(audio_id)
        if not db_doc:
            continue
        doc_vis = (db_doc.get("visibility") or "").strip().lower()
        if doc_vis not in allowed_vis_set:
            continue

        text = (doc.page_content or "").strip()
        citations.append(
            AudioCitation(
                audio_id=audio_id,
                segment_id=segment_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                clip_url=_clip_url(base, audio_id, start_ms, end_ms),
                score=float(score) if score is not None else None,
            )
        )
        if len(citations) >= k:
            break

    if not citations:
        return AudioAskResp(question=question, answer="没有检索到相关音频片段。", citations=[])

    api_key = getattr(settings, "qianwen_api_key", "") or ""
    model = getattr(settings, "model_name", "") or "gpt-4o-mini"

    if not api_key:
        return AudioAskResp(
            question=question,
            answer="(未配置 OPENAI_API_KEY) 已返回相关音频片段引用，可先基于citations手动判断。",
            citations=citations,
        )


    messages = _build_rag_messages(question, citations, req.system_prompt)
    answer = _openai_chat_complete(model=model, api_key=api_key, messages=messages, timeout_s=90.0)

    return AudioAskResp(question=question, answer=answer, citations=citations)
