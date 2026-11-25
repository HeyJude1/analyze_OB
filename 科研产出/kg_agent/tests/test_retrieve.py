"""检索测试：7 步 trace、dropped 原因、硬件过滤、条件求值、trace 落盘。"""

import unittest

from tests.helpers import built

from kgagent import graph as graph_mod
from kgagent import retrieve as retrieve_mod


class RetrieveTraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.data_dir, _ = built()
        cls.graph = graph_mod.load_graph(cls.data_dir)

    def test_trace_has_exactly_seven_steps_in_order(self):
        trace = retrieve_mod.retrieve(self.cfg, "dgemm", "x86_avx2", top_k=6,
                                      data_dir=self.data_dir, write=False)
        steps = trace["steps"]
        self.assertGreaterEqual(len(steps), 7)
        self.assertEqual([s["step"] for s in steps], list(range(1, len(steps) + 1)))
        expected = [name for name, _ in retrieve_mod.TRACE_STEPS]
        self.assertEqual([s["name"] for s in steps], expected)

    def test_every_step_has_required_fields_and_dropped_reason(self):
        trace = retrieve_mod.retrieve(self.cfg, "dgemm", "x86_avx2", data_dir=self.data_dir, write=False)
        for step in trace["steps"]:
            for field in ("name", "description", "candidates", "selected"):
                self.assertIn(field, step, "step %s 缺字段 %s" % (step["name"], field))
            for item in step.get("dropped") or []:
                self.assertIn("id", item)
                self.assertTrue(item.get("reason"), "淘汰项必须有 reason：%r" % item)

    def test_condition_and_hardware_filters_really_drop(self):
        trace = retrieve_mod.retrieve(self.cfg, "dgemm", "x86_avx2", data_dir=self.data_dir, write=False)
        by_name = {s["name"]: s for s in trace["steps"]}
        self.assertGreater(len(by_name["condition_filter"]["dropped"]), 0)
        self.assertGreater(len(by_name["hardware_filter"]["dropped"]), 0)
        # x86 profile 上 RVV 策略必须被硬件过滤掉
        hw_dropped = " ".join(item["reason"] for item in by_name["hardware_filter"]["dropped"])
        self.assertIn("x86_avx2", hw_dropped)

    def test_selection_respects_top_k_and_result_pack(self):
        trace = retrieve_mod.retrieve(self.cfg, "dgemm", "x86_avx2", top_k=3,
                                      data_dir=self.data_dir, write=False)
        result = trace["result"]
        self.assertLessEqual(len(result["strategies"]), 3)
        self.assertEqual(result["pack"]["schema_version"], "pack-v1")
        self.assertTrue(result["prompt_preview"])
        for strategy in result["pack"]["strategies"]:
            self.assertIn("score_breakdown", strategy)
            self.assertIn("why_selected", strategy)

    def test_rvv_profile_selects_rvv_strategies(self):
        trace = retrieve_mod.retrieve(self.cfg, "dgemm", "k1_rvv256", data_dir=self.data_dir, write=False)
        selected = trace["result"]["strategies"]
        self.assertTrue(selected)
        rvv = [sid for sid in selected if sid.endswith("_d")]
        self.assertTrue(rvv, "k1_rvv256 应能选到差分抽取的 RVV 策略实例")

    def test_trace_written_to_disk(self):
        trace = retrieve_mod.retrieve(self.cfg, "ddot", "x86_avx2", data_dir=self.data_dir, write=True)
        self.assertTrue(trace.get("trace_path"))
        import json
        from pathlib import Path
        path = Path(trace["trace_path"])
        self.assertTrue(path.exists())
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["trace_id"], trace["trace_id"])
        self.assertNotIn(":", path.name, "Windows 文件名不能包含 ':'")


class ConditionEvalTest(unittest.TestCase):
    def test_dtype_in_operator(self):
        result = retrieve_mod.evaluate_conditions(
            [{"pred": "dtype", "op": "in", "value": ["d"], "hard": True}], {"dtype": "d"})
        self.assertEqual(result["hard_fail"], [])
        self.assertEqual(result["fit"], 1.0)
        result = retrieve_mod.evaluate_conditions(
            [{"pred": "dtype", "op": "in", "value": ["c"], "hard": True}], {"dtype": "d"})
        self.assertTrue(result["hard_fail"])

    def test_loop_depth_ge_and_missing_feature(self):
        result = retrieve_mod.evaluate_conditions(
            [{"pred": "loop_depth", "op": "ge", "value": 2, "hard": True}], {"loop_depth": 3})
        self.assertEqual(result["hard_fail"], [])
        result = retrieve_mod.evaluate_conditions(
            [{"pred": "loop_depth", "op": "ge", "value": 2, "hard": True}], {"loop_depth": 1})
        self.assertTrue(result["hard_fail"])
        result = retrieve_mod.evaluate_conditions(
            [{"pred": "structural", "op": "eq", "value": "none", "hard": True}], {})
        self.assertTrue(result["hard_fail"], "硬条件遇到缺失特征应视为不满足")

    def test_soft_condition_contributes_to_fit(self):
        conditions = [
            {"pred": "dtype", "op": "in", "value": ["d"], "hard": True},
            {"pred": "aligned", "op": "eq", "value": True, "hard": False, "weight": 1.0},
        ]
        result = retrieve_mod.evaluate_conditions(conditions, {"dtype": "d", "aligned": False})
        self.assertEqual(result["hard_fail"], [])
        self.assertAlmostEqual(result["fit"], 0.5)


if __name__ == "__main__":
    unittest.main()
