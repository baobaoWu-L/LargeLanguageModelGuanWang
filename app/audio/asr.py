from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel


@dataclass		# 数据类
class ASRSegment:
    start_s: float  	# 开始时间
    end_s: float	# 结束时间
    text: str		# 转成文字后的文本


class ASR:
    def __init__(self, model_name='/home/lovebreaker/.cache/huggingface/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66',
                 device: str = "cpu", compute_type: str = "int8"):
        # 我是m2的芯片，可以考虑device="cpu", compute_type="int8"这样比较省资源，有显卡可以device="cuda"，compute_type="float16"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> tuple[List[ASRSegment], Optional[str]]:
        segments_iter, info = self.model.transcribe(
            wav_path,
            language=language,
            vad_filter=False,  # v0先关，后续v1加VAD
            initial_prompt='请用简体中文输出'
        )
        segs: List[ASRSegment] = []
        for s in segments_iter:
            txt = (s.text or "").strip()
            segs.append(ASRSegment(start_s=float(s.start), end_s=float(s.end), text=txt))
        lang = getattr(info, "language", None)

        return segs, lang
if __name__ == "__main__":
    segs, lang = ASR().transcribe("/app/chushibiao.wav", language="zh")

    print('这段音频中说的语言是:', lang)
    for s in segs:
        print(s)





