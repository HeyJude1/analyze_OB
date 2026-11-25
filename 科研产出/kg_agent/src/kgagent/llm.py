"""LLMClient：可选 API + 全量缓存 + offline 降级（SPEC §4 CLI build --llm）。

关键设计（方案 §5.7）：
1. 每次调用以 sha256(model + prompt_version + prompt) 为键落盘到 cache 目录；
2. offline 模式只读缓存，未命中直接抛 OfflineLLMUnavailable（**绝不偷偷联网**）；
3. API 模式用标准库 urllib 调 OpenAI 兼容端点（默认 DashScope compatible-mode），失败自动降级；
4. LLM 只被 S3 用来改写 name/description/rationale，结构字段（条件/参数/行号/数字）不经过 LLM。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__all__ = ["LLMClient", "OfflineLLMUnavailable"]


class OfflineLLMUnavailable(RuntimeError):
    """offline 模式下缓存未命中。"""


class LLMClient:
    def __init__(self, cfg, backend: str | None = None):
        llm_cfg = cfg.llm_cfg
        self.cfg = cfg
        self.backend = (backend or llm_cfg.get("backend") or "offline").lower()
        self.model = llm_cfg.get("model") or "qwen-plus"
        self.api_base = (llm_cfg.get("api_base") or "").rstrip("/")
        self.api_key_env = llm_cfg.get("api_key_env") or "DASHSCOPE_API_KEY"
        self.timeout_s = int(llm_cfg.get("timeout_s") or 60)
        self.prompt_version = cfg.prompt_version
        cache_dir = llm_cfg.get("cache_dir") or "data/llm_cache"
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = cfg.root / cache_path
        self.cache_dir = cache_path

    # ---- 缓存 ----
    def cache_key(self, prompt: str) -> str:
        raw = "%s|%s|%s" % (self.model, self.prompt_version, prompt)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / ("%s.json" % key)

    def cached(self, prompt: str):
        path = self._cache_path(self.cache_key(prompt))
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _store(self, prompt: str, payload) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(self.cache_key(prompt))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 调用 ----
    def generate_json(self, prompt: str, purpose: str = "s3", schema: dict | None = None) -> dict:
        cached = self.cached(prompt)
        if cached is not None:
            return cached
        if self.backend == "offline":
            raise OfflineLLMUnavailable(
                "offline 模式无缓存：purpose=%s key=%s。请先联网跑 build --llm 生成缓存，或使用 build --offline"
                % (purpose, self.cache_key(prompt)[:12]))
        payload = self._call_api(prompt)
        self._store(prompt, payload)
        return payload

    def _call_api(self, prompt: str) -> dict:
        # 只有显式 --llm + backend=api 才会走网络；import 放在函数内，保证 offline 零网络依赖
        import urllib.request

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError("环境变量 %s 未设置，无法调用 LLM；请改用 build --offline" % self.api_key_env)
        if not self.api_base:
            raise RuntimeError("config.yaml 未配置 llm.api_base")
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "你是 HPC 算子优化知识抽取器，只输出 JSON，不输出任何性能数字。"},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer %s" % api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        return json.loads(content)
