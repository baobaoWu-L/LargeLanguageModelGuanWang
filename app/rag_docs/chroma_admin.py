# from __future__ import annotations
#
# from typing import Any
# import chromadb
# from app.config import settings
#
#
# def get_collection():
#     client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
#     return client.get_or_create_collection(settings.collection_name)
#
#
# def delete_by_doc_id(doc_id: str) -> int:
#     # chromadb特有的api用来按照doc_id删除其中的一个文档
#     col = get_collection()  # 这个函数用于获取我向量数据中集合的名字
#     try:
#         before = col.count()
#         col.delete(where={"doc_id": doc_id})
#         after = col.count()
#         return int(before - after)
#     except Exception:
#         got = col.get(where={"doc_id": doc_id})
#         ids = got.get("ids") or []
#         if ids:
#             col.delete(ids=ids)
#         return len(ids)
#
#
# def get_ids_and_metadatas_by_doc_id(doc_id: str) -> tuple[list[str], list[dict[str, Any]]]:
#     col = get_collection()
#     got = col.get(where={"doc_id": doc_id}, include=["metadatas"])
#     ids = got.get("ids") or []
#     metas = got.get("metadatas") or []
#     return list(ids), list(metas)
#
#
# def count_by_doc_id(doc_id: str) -> int:
#     ids, _ = get_ids_and_metadatas_by_doc_id(doc_id)
#     return len(ids)
#
#
# def update_visibility_by_doc_id(doc_id: str, visibility: str) -> int:
#     # 更新文档的可见性
#     col = get_collection()
#     ids, metas = get_ids_and_metadatas_by_doc_id(doc_id)
#     if not ids:
#         return 0
#     new_metas: list[dict[str, Any]] = []
#     for m in metas:
#         mm = dict(m or {})
#         mm["visibility"] = visibility
#         new_metas.append(mm)
#     col.update(ids=ids, metadatas=new_metas)
#     return len(ids)
#
# def delete_by_audio_id(audio_id: str) -> int:
#     col = get_collection()
#     try:
#         before = col.count()
#         col.delete(where={"audio_id": audio_id})
#         after = col.count()
#         return max(0, int(before - after))
#     except Exception:
#         got = col.get(where={"audio_id": audio_id})
#         ids = got.get("ids") or []
#         if ids:
#             col.delete(ids=ids)
#         return len(ids)

# 修改点：把delete_by_audio_id()从KB collection改为audio collection
# 新增点：reset_audio_collection()、update_visibility_by_audio_id()、批量删除等，主要是为了管理台需要
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import chromadb

from app.config import settings


@lru_cache(maxsize=1)  # 缓存最近1个
def _client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


@lru_cache(maxsize=8)  # 缓存最近8个
def _collection(name: str):
    return _client().get_or_create_collection(name)


def get_kb_collection():
    return _collection(settings.collection_name)


def get_collection():
    return get_kb_collection()


def delete_by_doc_id(doc_id: str) -> int:
    col = get_kb_collection()
    try:
        before = col.count()
        col.delete(where={"doc_id": doc_id})
        after = col.count()
        return max(0, int(before - after))
    except Exception:
        got = col.get(where={"doc_id": doc_id})
        ids = got.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)


def get_ids_and_metadatas_by_doc_id(doc_id: str) -> tuple[list[str], list[dict[str, Any]]]:
    col = get_kb_collection()
    got = col.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    return list(ids), list(metas)


def count_by_doc_id(doc_id: str) -> int:
    ids, _ = get_ids_and_metadatas_by_doc_id(doc_id)
    return len(ids)


def update_visibility_by_doc_id(doc_id: str, visibility: str) -> int:
    col = get_kb_collection()
    ids, metas = get_ids_and_metadatas_by_doc_id(doc_id)
    if not ids:
        return 0
    new_metas: list[dict[str, Any]] = []
    for m in metas:
        mm = dict(m or {})
        mm["visibility"] = visibility
        new_metas.append(mm)
    col.update(ids=ids, metadatas=new_metas)
    return len(ids)


def reset_kb_collection() -> None:
    try:
        _client().delete_collection(settings.collection_name)
    except Exception:
        pass
    _collection.cache_clear()
    get_kb_collection()


def get_audio_collection():
    return _collection(settings.audio_collection_name)


def delete_by_audio_id(audio_id: str) -> int:
    col = get_audio_collection()
    try:
        before = col.count()
        col.delete(where={"audio_id": audio_id})
        after = col.count()
        return max(0, int(before - after))
    except Exception:
        got = col.get(where={"audio_id": audio_id})
        ids = got.get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)


def update_visibility_by_audio_id(audio_id: str, visibility: str) -> int:
    col = get_audio_collection()
    got = col.get(where={"audio_id": audio_id}, include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    if not ids:
        return 0
    new_metas: list[dict[str, Any]] = []
    for m in metas:
        mm = dict(m or {})
        mm["visibility"] = visibility
        new_metas.append(mm)
    col.update(ids=ids, metadatas=new_metas)
    return len(ids)


def delete_many_audio_ids(audio_ids: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for aid in audio_ids:
        aid = (aid or "").strip()
        if not aid:
            continue
        try:
            out[aid] = int(delete_by_audio_id(aid))
        except Exception:
            out[aid] = 0
    return out


def reset_audio_collection() -> None:
    try:
        _client().delete_collection(settings.audio_collection_name)
    except Exception:
        pass
    _collection.cache_clear()
    get_audio_collection()
