"""
反馈机制 — 检索结果反馈闭环
记录检索结果, 更新原则 evidence_strength, 检测冲突/组合信号
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from .vector_store import VectorStore


class Feedback:
    def __init__(self, store: VectorStore, feedback_file: str = None):
        self.store = store
        self.feedback_file = feedback_file or "output/feedback_log.jsonl"

    def record(self, code_hash: str, extracted_chars: List[Dict],
               recommendations: List[Dict], selected_uid: Optional[str] = None,
               outcome: str = "unknown", outcome_detail: str = ""):
        """记录一次检索事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "code_hash": code_hash,
            "extracted_chars": [c.get("characteristic_type", "") for c in extracted_chars],
            "recommended_uids": [r.get("principle_uid", "") for r in recommendations[:10]],
            "recommended_scores": [r.get("final_score", 0) for r in recommendations[:10]],
            "selected_uid": selected_uid,
            "outcome": outcome,  # "applied_success", "applied_no_effect", "not_applied", "unknown"
            "outcome_detail": outcome_detail
        }
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    def update_evidence(self, principle_uid: str) -> float:
        """基于反馈历史更新原则的 evidence_strength"""
        events = self._load_events_for_principle(principle_uid)
        if len(events) < 3:
            return 0.5  # 不足, 保持默认

        total = len(events)
        successes = sum(1 for e in events if e.get("outcome") == "applied_success")
        failures = sum(1 for e in events if e.get("outcome") == "applied_no_effect")

        # 指数衰减: 新反馈权重更高
        now = time.time()
        weighted_success = 0.0
        weighted_total = 0.0
        for e in events:
            age_days = (now - time.mktime(
                time.strptime(e["timestamp"][:10], "%Y-%m-%d"))) / 86400
            weight = max(0.1, 2.0 ** (-age_days / 30))  # 30天半衰期
            weighted_total += weight
            if e.get("outcome") == "applied_success":
                weighted_success += weight

        new_strength = round(weighted_success / weighted_total, 3) if weighted_total > 0 else 0.5
        print(f"  📊 {principle_uid[:8]}: evidence {0.5:.2f} → {new_strength:.2f} "
              f"({successes}✓/{failures}✗/{total}total)")
        return new_strength

    def detect_signals(self) -> List[Dict]:
        """检测新关系信号 (COMPOSES_WITH / CONFLICTS_WITH)"""
        events = self._load_all_events()
        if len(events) < 10:
            return []

        # 共现分析: 经常一起被推荐且都成功的 → 建议 COMPOSES_WITH
        co_occurrence = {}
        for e in events:
            uids = e.get("recommended_uids", [])
            if e.get("outcome") == "applied_success":
                for i, a in enumerate(uids):
                    for b in uids[i + 1:]:
                        key = f"{a}|{b}"
                        co_occurrence[key] = co_occurrence.get(key, 0) + 1

        signals = []
        for key, count in co_occurrence.items():
            if count >= 3:
                a, b = key.split("|")
                signals.append({
                    "type": "suggest_COMPOSES_WITH",
                    "principle_a": a, "principle_b": b,
                    "co_occurrence_count": count, "confidence": min(count / 10, 1.0)
                })
        return signals

    def _load_events_for_principle(self, uid: str) -> List[Dict]:
        events = self._load_all_events()
        return [e for e in events if uid in e.get("recommended_uids", [])]

    def _load_all_events(self) -> List[Dict]:
        try:
            with open(self.feedback_file) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []
