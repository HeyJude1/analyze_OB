"""MPSA（B3/B4）测试：mock 下产出合法 JSON、代码清洗、无 key 报错、CLI 端到端。

全部离线：mock provider 不联网、不需要 DASHSCOPE_API_KEY；真实 API 由 Lead 跑。
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import built

from kgagent import cli
from kgagent import mpsa

DAXPY_GENERIC = """\
#include <stddef.h>

typedef long long BLASLONG;

int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;

 if ( n < 0 ) return(0);
 if ( da == 0.0 ) return(0);

 while(i < n)
 {
  y[iy] += da * x[ix] ;
  ix += inc_x ;
  iy += inc_y ;
  i++ ;
 }
 return(0);
}
"""

DDOT_GENERIC = """\
#include <stddef.h>

typedef long long BLASLONG;

double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;
 double dot = 0.0 ;

 if ( n < 0 ) return(dot);

 while(i < n)
 {
  dot += y[iy] * x[ix];
  ix += inc_x ;
  iy += inc_y ;
  i++ ;
 }
 return(dot);
}
"""

RESULT_KEYS = {"op", "profile", "mode", "model", "retrieved", "analysis", "synthesis",
               "prompt", "code", "usage", "elapsed_s"}


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def strip_blank_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.strip())


class MPSATestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.data_dir, _ = built()
        cls.tmp = Path(tempfile.mkdtemp(prefix="mpsa_test_"))
        cls.daxpy_code = write_text(cls.tmp / "daxpy_generic.c", DAXPY_GENERIC)
        cls.ddot_code = write_text(cls.tmp / "ddot_generic.c", DDOT_GENERIC)

    def run_mock(self, op="daxpy", mode="b3", code_path=None, out_name=None, **kwargs):
        out_dir = self.tmp / (out_name or ("out_%s_%s" % (op, mode)))
        return mpsa.run_mpsa(
            self.cfg, op, "k1_rvv256", mode=mode, provider="mock",
            code_path=str(code_path or self.daxpy_code), out_dir=str(out_dir),
            data_dir=self.data_dir, **kwargs,
        )


class MockProviderTest(MPSATestBase):
    def test_mock_b3_result_json_and_generated_c(self):
        result = self.run_mock(mode="b3")
        self.assertEqual(result["schema_version"], "mpsa-v1")
        self.assertTrue(RESULT_KEYS.issubset(result.keys()))
        self.assertEqual(result["mode"], "b3")
        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["model"], "mock")
        self.assertGreater(result["retrieved"]["count"], 0)
        self.assertTrue(result["retrieved"]["strategy_ids"])
        self.assertIsNone(result["analysis"])
        self.assertIsNone(result["synthesis"])
        self.assertIn("daxpy_k(", result["code"])
        self.assertNotIn("```", result["code"])
        self.assertTrue(result["code"].rstrip().endswith("}"))
        self.assertEqual(result["usage"]["calls"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 0)

        result_path = Path(result["artifacts"]["result_path"])
        generated_path = Path(result["artifacts"]["generated_c_path"])
        self.assertTrue(result_path.exists())
        self.assertTrue(generated_path.exists())
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["code"], result["code"])
        self.assertEqual(generated_path.read_text(encoding="utf-8"), result["code"])
        self.assertEqual(saved["mode"], "b3")

    def test_mock_b4_analysis_and_synthesis(self):
        result = self.run_mock(mode="b4", analysis_k=2)
        self.assertEqual(result["mode"], "b4")
        analysis = result["analysis"]
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis["items"]), 2)
        for item in analysis["items"]:
            for key in ("strategy_id", "applies", "evidence", "risks", "parameters", "conflicts"):
                self.assertIn(key, item, "分析项缺字段 %s" % key)
            self.assertIsInstance(item["applies"], bool)
            self.assertIsInstance(item["evidence"], list)
        synthesis = result["synthesis"]
        self.assertIsNotNone(synthesis)
        self.assertEqual(set(synthesis["criteria_scores"]), set(mpsa.CRITERIA))
        for score in synthesis["criteria_scores"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        self.assertTrue(synthesis["overall"])
        # b4 = analysis_k 次分析 + 1 次综合 + 1 次生成
        self.assertEqual(result["usage"]["calls"], 2 + 2)
        self.assertEqual(result["usage"]["by_stage"]["analysis"]["calls"], 2)
        self.assertEqual(result["usage"]["by_stage"]["synthesis"]["calls"], 1)
        for key in ("analysis", "synthesis", "generate"):
            self.assertTrue(result["prompt"][key])
        self.assertEqual(len(result["prompt"]["analysis"]), 2)

    def test_mock_is_deterministic(self):
        first = self.run_mock(mode="b4", out_name="det_a")
        second = self.run_mock(mode="b4", out_name="det_b")
        self.assertEqual(first["analysis"], second["analysis"])
        self.assertEqual(first["synthesis"], second["synthesis"])
        self.assertEqual(first["code"], second["code"])
        self.assertEqual(first["retrieved"]["strategy_ids"], second["retrieved"]["strategy_ids"])

    def test_mock_ddot_uses_generic_signature(self):
        result = self.run_mock(op="ddot", mode="b3", code_path=self.ddot_code, out_name="ddot_b3")
        self.assertIn("double ddot_k(", result["code"])
        self.assertNotIn("daxpy", result["code"])
        self.assertTrue(result["code_meta"]["prologue_added"])

    def test_mock_code_file_override(self):
        mock_code = write_text(
            self.tmp / "mock_code.c",
            "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
            "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)\n"
            "{\n    /* MOCK_CODE_MARKER */\n    return 0;\n}\n",
        )
        result = self.run_mock(mode="b3", mock_code_path=str(mock_code), out_name="mock_code")
        self.assertIn("MOCK_CODE_MARKER", result["code"])
        self.assertIn("daxpy_k(", result["code"])

    def test_mock_unknown_op_without_code_raises_clear_error(self):
        with self.assertRaises(mpsa.MPSAConfigError) as ctx:
            mpsa.run_mpsa(
                self.cfg, "dgemm", "x86_avx2", mode="b3", provider="mock", top_k=2,
                out_dir=str(self.tmp / "nope"), data_dir=self.data_dir,
            )
        self.assertIn("--code", str(ctx.exception))


class CleanCodeTest(unittest.TestCase):
    def test_strips_markdown_fence_and_prose(self):
        raw = (
            "下面是我生成的优化代码：\n"
            "```c\n"
            "#include <stddef.h>\n"
            "int daxpy_k(BLASLONG n, double da, double *x, double *y) {\n"
            "    (void)n; (void)da; (void)x; (void)y;\n"
            "    return 0;\n"
            "}\n"
            "```\n"
            "以上代码只做示例。\n"
        )
        cleaned = mpsa.clean_code(raw, op="daxpy", canonical="axpy")
        self.assertTrue(cleaned["fences_removed"])
        self.assertNotIn("```", cleaned["code"])
        self.assertNotIn("下面是我生成的优化代码", cleaned["code"])
        self.assertNotIn("以上代码只做示例", cleaned["code"])
        self.assertIn("int daxpy_k(", cleaned["code"])
        self.assertIn("#include <stddef.h>", cleaned["code"])
        self.assertTrue(cleaned["code"].endswith("\n"))

    def test_wraps_body_only_with_task_signature(self):
        raw = "for (i = 0; i < n; i++) { y[i] += da * x[i]; }\nreturn 0;"
        cleaned = mpsa.clean_code(raw, op="daxpy", canonical="axpy")
        self.assertTrue(cleaned["wrapped_body"])
        self.assertIn("int daxpy_k(", cleaned["code"])
        self.assertIn("typedef long long BLASLONG;", cleaned["code"])
        self.assertIn("y[i] += da * x[i];", cleaned["code"])
        depth, negative = mpsa._brace_depth(cleaned["code"])
        self.assertFalse(negative)
        self.assertEqual(depth, 0)

    def test_ddot_body_only_uses_builtin_signature(self):
        raw = "double acc = 0.0;\nacc += x[0] * y[0];\nreturn acc;"
        cleaned = mpsa.clean_code(raw, op="ddot", canonical="dot")
        self.assertTrue(cleaned["wrapped_body"])
        self.assertIn("double ddot_k(", cleaned["code"])
        self.assertIn("return acc;", cleaned["code"])

    def test_renames_renamed_function_back_to_task_name(self):
        raw = (
            "int daxpy_opt(BLASLONG n, double da, double *x, double *y) {\n"
            "    (void)n; (void)da; (void)x; (void)y;\n"
            "    return 0;\n"
            "}\n"
        )
        signature = "int daxpy_k(BLASLONG n, double da, double *x, double *y)"
        cleaned = mpsa.clean_code(raw, op="daxpy", canonical="axpy", signature=signature)
        self.assertIn("int daxpy_k(", cleaned["code"])
        self.assertNotIn("daxpy_opt", cleaned["code"])
        self.assertTrue(any("renamed" in item for item in cleaned["repairs"]))

    def test_keeps_static_helper_before_target_function(self):
        raw = (
            "#include <stddef.h>\n"
            "\n"
            "static double clamp(double v) { return v > 0.0 ? v : 0.0; }\n"
            "\n"
            "int daxpy_k(BLASLONG n, double da, double *x, double *y)\n"
            "{\n"
            "    for (BLASLONG i = 0; i < n; i++) y[i] += clamp(da) * x[i];\n"
            "    return 0;\n"
            "}\n"
        )
        signature = "int daxpy_k(BLASLONG n, double da, double *x, double *y)"
        cleaned = mpsa.clean_code(raw, op="daxpy", canonical="axpy", signature=signature)
        self.assertIn("static double clamp", cleaned["code"])
        self.assertIn("clamp(da)", cleaned["code"])
        self.assertIn("#include <stddef.h>", cleaned["code"])
        self.assertEqual(mpsa._brace_depth(cleaned["code"]), (0, False))

    def test_extract_signature_prefers_generic_c(self):
        signature = mpsa.extract_signature(DDOT_GENERIC)
        self.assertIn("ddot_k", signature)
        self.assertIn("double", signature)
        parsed = mpsa.parse_signature(signature)
        self.assertEqual(parsed["name"], "ddot_k")
        self.assertEqual(len(parsed["params"]), 5)


class ErrorHandlingTest(MPSATestBase):
    def test_missing_api_key_is_clear_and_never_calls_network(self):
        env = dict(os.environ)
        env.pop("DASHSCOPE_API_KEY", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(mpsa.MissingAPIKeyError) as ctx:
                mpsa.run_mpsa(
                    self.cfg, "daxpy", "k1_rvv256", mode="b3", provider="dashscope",
                    code_path=str(self.daxpy_code), out_dir=str(self.tmp / "no_key"),
                    data_dir=self.data_dir,
                )
        message = str(ctx.exception)
        self.assertIn("DASHSCOPE_API_KEY", message)
        self.assertIn("mock", message)
        self.assertEqual(ctx.exception.exit_code, 3)

    def test_unknown_op_raises_retrieval_error(self):
        with self.assertRaises(mpsa.RetrievalError) as ctx:
            mpsa.run_mpsa(
                self.cfg, "not_an_op", "k1_rvv256", mode="b3", provider="mock",
                out_dir=str(self.tmp / "bad_op"), data_dir=self.data_dir,
            )
        self.assertEqual(ctx.exception.exit_code, 5)
        self.assertIn("检索失败", str(ctx.exception))

    def test_invalid_mode_and_provider(self):
        with self.assertRaises(mpsa.MPSAConfigError):
            mpsa.run_mpsa(self.cfg, "daxpy", "k1_rvv256", mode="b9", provider="mock",
                          data_dir=self.data_dir, write=False)
        with self.assertRaises(mpsa.MPSAConfigError):
            mpsa.run_mpsa(self.cfg, "daxpy", "k1_rvv256", mode="b3", provider="openai",
                          data_dir=self.data_dir, write=False)


class FakeDashScopeTest(MPSATestBase):
    """不打网络：用假 client 验证 dashscope 代码路径（JSON 解析 / usage 累加 / 代码清洗）。"""

    def test_b4_with_fake_dashscope_client(self):
        stages = []
        fake_function = (
            "```c\n"
            "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
            "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)\n"
            "{\n"
            "    (void)d0; (void)d1; (void)dummy; (void)d2;\n"
            "    for (BLASLONG i = 0; i < n; i++) y[i * inc_y] += da * x[i * inc_x];\n"
            "    return 0;\n"
            "}\n"
            "```"
        )

        def fake_chat(self, prompt, system, json_mode=False, stage="generate",
                      temperature=0.0, max_tokens=None):
            stages.append(stage)
            if stage == "analysis":
                payload = {"applies": True, "confidence": 0.7, "evidence": ["条件匹配"],
                           "risks": ["未实测"], "parameters": {"lmul": 2}, "conflicts": []}
                return json.dumps(payload), {"prompt_tokens": 10, "completion_tokens": 5,
                                             "total_tokens": 15}
            if stage == "synthesis":
                payload = {
                    "criteria": {key: {"score": 0.8, "reason": "ok"} for key in mpsa.CRITERIA},
                    "overall": "可以组合", "recommended": ["stg:axpy:vectorize_d"],
                    "guidance": "用 vsetvl 主循环 + 掩码尾部", "conflicts": [],
                }
                return json.dumps(payload), {"prompt_tokens": 20, "completion_tokens": 10,
                                             "total_tokens": 30}
            return fake_function, {"prompt_tokens": 100, "completion_tokens": 200,
                                   "total_tokens": 300}

        env = {"DASHSCOPE_API_KEY": "dummy-key-for-offline-test"}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            mpsa.DashScopeClient, "chat", fake_chat
        ):
            result = mpsa.run_mpsa(
                self.cfg, "daxpy", "k1_rvv256", mode="b4", provider="dashscope",
                code_path=str(self.daxpy_code), analysis_k=2,
                out_dir=str(self.tmp / "fake_api"), data_dir=self.data_dir,
            )
        self.assertEqual(result["provider"], "dashscope")
        self.assertEqual(result["model"], mpsa.DEFAULT_MODEL)
        self.assertEqual(sorted(stages), ["analysis", "analysis", "generate", "synthesis"])
        self.assertEqual(result["usage"]["calls"], 4)
        self.assertEqual(result["usage"]["total_tokens"], 2 * 15 + 30 + 300)
        self.assertEqual(result["usage"]["by_stage"]["generate"]["completion_tokens"], 200)
        self.assertEqual(len(result["analysis"]["items"]), 2)
        self.assertEqual(result["analysis"]["items"][0]["parameters"], {"lmul": 2})
        self.assertEqual(set(result["synthesis"]["criteria_scores"]), set(mpsa.CRITERIA))
        self.assertNotIn("```", result["code"])
        self.assertIn("for (BLASLONG i = 0; i < n; i++)", result["code"])

    def test_normalize_synthesis_rejects_missing_criteria(self):
        with self.assertRaises(mpsa.LLMError) as ctx:
            mpsa.normalize_synthesis({"criteria": {"supportive": {"score": 0.5, "reason": "x"}}})
        self.assertIn("complementarity", str(ctx.exception))


class RepairTest(MPSATestBase):
    """修复回路（mpsa-repair / repair_code）：mock 合法代码、错误输入、缺 key、dashscope 假 client。"""

    BROKEN_DAXPY = (
        "#include <riscv_vector.h>\n"
        "typedef long long BLASLONG;\n"
        "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
        "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)\n"
        "{\n"
        "    (void)d0; (void)d1; (void)dummy; (void)d2;\n"
        "    size_t vl = __riscv_vsetvl_e64m8(n);\n"
        "    vfloat64m8_t vx = __riscv_vlseg1e64_v_f64m8(x, vl);  /* 幻觉 intrinsic */\n"
        "    (void)vx; (void)da; (void)inc_x; (void)inc_y; (void)y;\n"
        "    return 0;\n"
        "}\n"
    )

    BROKEN_DDOT = (
        "#include <riscv_vector.h>\n"
        "typedef long long BLASLONG;\n"
        "double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)\n"
        "{\n"
        "    size_t vl = __riscv_vsetvl_e64m8(n);\n"
        "    vfloat64m8_t vacc = __riscv_vmvinf_f64m8();\n"
        "    (void)vl; (void)vacc; (void)x; (void)y; (void)inc_x; (void)inc_y;\n"
        "    return 0.0;  /* wrong_answer：只返回部分和 */\n"
        "}\n"
    )

    COMPILE_ERROR = (
        "generated.c: In function 'daxpy_k':\n"
        "generated.c:24:31: warning: implicit declaration of function '__riscv_vlseg1e64_v_f64m8';"
        " did you mean '__riscv_vlse64_v_f64m8'? [-Wimplicit-function-declaration]\n"
        "generated.c:24:31: error: incompatible types when initializing type 'vfloat64m8_t'"
        " using type 'int'\n"
    )

    WRONG_ANSWER = (
        "ddot n=16384 maxdiff=7.932e+03 reldiff=9.688e-01 best=0.178223 ms (1.47 GB/s)\n"
    )

    def repair(self, code, error, provider="mock", out_name="repair", **kwargs):
        return mpsa.repair_code(
            self.cfg, "daxpy", "k1_rvv256", code, error, mode="b3", provider=provider,
            out_dir=str(self.tmp / out_name), **kwargs,
        )

    def test_mock_repair_returns_valid_code_and_result_json(self):
        code_path = write_text(self.tmp / "broken_daxpy.c", self.BROKEN_DAXPY)
        error_path = write_text(self.tmp / "build.stderr.log", self.COMPILE_ERROR)
        result = self.repair(str(code_path), str(error_path), out_name="repair_axpy")
        self.assertEqual(result["schema_version"], "mpsa-repair-v1")
        for key in ("code", "usage", "code_meta", "input_error", "error_source",
                    "fix_summary", "elapsed_s", "calls", "prompt", "artifacts"):
            self.assertIn(key, result)
        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["model"], "mock")
        self.assertEqual(result["error_source"], str(error_path))
        self.assertIn("vlseg1e64", result["input_error"])
        self.assertIn("daxpy_k(", result["code"])
        self.assertNotIn("vlseg1e64", result["code"], "mock 修复应退回标量实现")
        self.assertNotIn("```", result["code"])
        self.assertEqual(result["usage"]["calls"], 1)
        self.assertEqual(result["usage"]["by_stage"]["repair"]["calls"], 1)
        self.assertEqual(result["usage"]["total_tokens"], 0)
        self.assertTrue(result["fix_summary"]["changed"])
        self.assertTrue(result["fix_summary"]["mock"])
        self.assertEqual(result["code_meta"]["function_name"], "daxpy_k")
        result_path = Path(result["artifacts"]["result_path"])
        repaired_path = Path(result["artifacts"]["repaired_c_path"])
        self.assertTrue(result_path.exists())
        self.assertTrue(repaired_path.exists())
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["code"], result["code"])
        self.assertEqual(repaired_path.read_text(encoding="utf-8"), result["code"])

    def test_repair_accepts_inline_error_text_and_keeps_it_in_prompt(self):
        code_path = write_text(self.tmp / "broken_ddot.c", self.BROKEN_DDOT)
        result = mpsa.repair_code(
            self.cfg, "ddot", "k1_rvv256", str(code_path), self.WRONG_ANSWER,
            mode="b4", provider="mock", out_dir=str(self.tmp / "repair_inline"),
        )
        self.assertEqual(result["error_source"], "inline")
        self.assertIn("maxdiff=7.932e+03", result["prompt"]["repair"])
        self.assertIn("ddot_k(", result["code"])
        self.assertIn("return acc;", result["code"])
        self.assertNotIn("vmvinf", result["code"])
        self.assertEqual(result["mode"], "b4")

    def test_repair_rejects_empty_or_missing_inputs(self):
        code_path = write_text(self.tmp / "broken_bad_input.c", self.BROKEN_DAXPY)
        with self.assertRaises(mpsa.MPSAConfigError):
            self.repair(str(code_path), "", out_name="repair_empty_error")
        with self.assertRaises(mpsa.MPSAConfigError) as ctx:
            self.repair(str(code_path), r"C:\no_such_dir\build.stderr.log",
                        out_name="repair_missing_error_file")
        self.assertIn("文件不存在", str(ctx.exception))
        with self.assertRaises(mpsa.MPSAConfigError):
            self.repair("", self.COMPILE_ERROR, out_name="repair_empty_code")

    def test_repair_missing_api_key_is_clear(self):
        code_path = write_text(self.tmp / "broken_nokey.c", self.BROKEN_DAXPY)
        env = dict(os.environ)
        env.pop("DASHSCOPE_API_KEY", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(mpsa.MissingAPIKeyError) as ctx:
                self.repair(str(code_path), self.COMPILE_ERROR, provider="dashscope",
                            out_name="repair_nokey")
        self.assertIn("DASHSCOPE_API_KEY", str(ctx.exception))
        self.assertIn("mock", str(ctx.exception))
        self.assertEqual(ctx.exception.exit_code, 3)

    def test_mock_code_override_for_repair(self):
        code_path = write_text(self.tmp / "broken_override.c", self.BROKEN_DAXPY)
        override = write_text(
            self.tmp / "override_repaired.c",
            "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
            "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)\n"
            "{\n    /* REPAIR_OVERRIDE_MARKER */\n    (void)n; (void)d0; (void)d1; (void)da;\n"
            "    (void)x; (void)inc_x; (void)y; (void)inc_y; (void)dummy; (void)d2;\n"
            "    return 0;\n}\n",
        )
        result = self.repair(str(code_path), self.COMPILE_ERROR,
                             mock_code_path=str(override), out_name="repair_override")
        self.assertIn("REPAIR_OVERRIDE_MARKER", result["code"])

    def test_fake_dashscope_repair_path(self):
        code_path = write_text(self.tmp / "broken_fake_api.c", self.BROKEN_DAXPY)
        repaired = (
            "```c\n"
            "typedef long long BLASLONG;\n"
            "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
            "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)\n"
            "{\n    (void)d0; (void)d1; (void)dummy; (void)d2;\n"
            "    for (BLASLONG i = 0; i < n; i++) y[i * inc_y] += da * x[i * inc_x];\n"
            "    return 0;\n}\n```"
        )
        calls = []

        def fake_chat(self, prompt, system, json_mode=False, stage="generate",
                      temperature=0.0, max_tokens=None):
            calls.append((stage, max_tokens, prompt))
            return repaired, {"prompt_tokens": 120, "completion_tokens": 80,
                              "total_tokens": 200}

        env = {"DASHSCOPE_API_KEY": "dummy-key-for-offline-test"}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            mpsa.DashScopeClient, "chat", fake_chat
        ):
            result = self.repair(str(code_path), self.COMPILE_ERROR, provider="dashscope",
                                 out_name="repair_fake_api")
        self.assertEqual(result["provider"], "dashscope")
        self.assertEqual(result["model"], mpsa.DEFAULT_MODEL)
        self.assertEqual([call[0] for call in calls], ["repair"])
        self.assertEqual(result["usage"]["total_tokens"], 200)
        self.assertEqual(result["usage"]["by_stage"]["repair"]["completion_tokens"], 80)
        self.assertNotIn("```", result["code"])
        self.assertIn("for (BLASLONG i = 0; i < n; i++)", result["code"])
        self.assertIn("implicit declaration", result["prompt"]["repair"])

    def test_cli_mpsa_repair_mock(self):
        code_path = write_text(self.tmp / "broken_cli.c", self.BROKEN_DAXPY)
        error_path = write_text(self.tmp / "qemu.stdout.log", self.WRONG_ANSWER)
        out_dir = self.tmp / "cli_repair"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([
                "mpsa-repair", "--op", "daxpy", "--profile", "k1_rvv256",
                "--code", str(code_path), "--error", str(error_path),
                "--mode", "b3", "--provider", "mock", "--out", str(out_dir),
            ])
        self.assertEqual(code, 0, stderr.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["provider"], "mock")
        self.assertEqual(summary["calls"], 1)
        self.assertTrue(Path(summary["result_path"]).exists())
        self.assertTrue(Path(summary["repaired_c_path"]).exists())
        self.assertTrue((out_dir / "repair_result.json").exists())


class CliTest(MPSATestBase):
    def test_cli_mpsa_mock_exit_code_and_output(self):
        out_dir = self.tmp / "cli_mock"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([
                "mpsa", "--op", "daxpy", "--profile", "k1_rvv256",
                "--code", str(self.daxpy_code), "--mode", "b4",
                "--provider", "mock", "--out", str(out_dir),
                "--top-k", "4", "--analysis-k", "2",
                "--data-dir", str(self.data_dir),
            ])
        self.assertEqual(code, 0, stderr.getvalue())
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["mode"], "b4")
        result_path = Path(summary["result_path"])
        self.assertTrue(result_path.exists())
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(set(saved["synthesis"]["criteria_scores"]), set(mpsa.CRITERIA))
        self.assertTrue((out_dir / "generated.c").exists())

    def test_cli_missing_key_returns_exit_code_3(self):
        env = dict(os.environ)
        env.pop("DASHSCOPE_API_KEY", None)
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stderr(stderr):
                code = cli.main([
                    "mpsa", "--op", "daxpy", "--profile", "k1_rvv256",
                    "--code", str(self.daxpy_code), "--mode", "b3",
                    "--provider", "dashscope", "--out", str(self.tmp / "cli_nokey"),
                    "--data-dir", str(self.data_dir),
                ])
        self.assertEqual(code, 3)
        self.assertIn("DASHSCOPE_API_KEY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
