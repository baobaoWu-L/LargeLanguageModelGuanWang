from __future__ import annotations

from celery import Celery
from app.config import settings

celery_app = Celery("enterprise_kb_assistant", broker=settings.celery_broker_url)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    timezone="UTC",
)

celery_app.autodiscover_tasks(["app.tasks"])
# 向消息队列里发送一个消息且消息队列收到后，它会自动执行这个任务