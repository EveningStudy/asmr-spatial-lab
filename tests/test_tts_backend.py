import ast
import unittest
from pathlib import Path

from bridge import Installation


class TTSBackendTests(unittest.TestCase):
    def test_paths_select_20(self):
        installation = object.__new__(Installation)
        installation.repo = Path("fixture")
        self.assertEqual(installation.runtime, Path("fixture/.asmr-dubber/runtimes/index-tts"))

    def test_worker_uses_20_contract(self):
        tree = ast.parse((Path(__file__).parents[1] / "tts_worker.py").read_text(encoding="utf-8"))
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertIn("indextts.infer_v2", modules)
        self.assertNotIn("indextts.infer_v2_5", modules)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        constructor = next(
            n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "IndexTTS2"
        )
        keywords = {k.arg for k in constructor.keywords}
        self.assertIn("use_fp16", keywords)
        self.assertNotIn("use_bf16", keywords)
        forbidden = {"lang", "duration_factor", "use_qwen_emo"}
        self.assertFalse(forbidden & {k.arg for n in calls for k in n.keywords})


if __name__ == "__main__":
    unittest.main()
