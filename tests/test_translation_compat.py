import json
import sys
import unittest
from pathlib import Path

import httpx

from translation_compat import TranslationClient, normalize_content, translate_batches

sys.path.insert(0, str(Path(__file__).parents[2] / "asmr-next/src"))
from asmr_dubber.errors import TranslationError
from asmr_dubber.models import Sentence
from asmr_dubber.translation import DeepSeekTranslator, TranslationChunk, validate_translation


class TranslationCompatibilityTests(unittest.TestCase):
    def test_empty_retries_only_failed_sentence(self):
        segments = [{"id": "done", "zh": "已有译文"}, {"id": "a", "zh": ""}, {"id": "b", "zh": ""}]
        calls, checkpoints = [], []

        def request(batch):
            calls.append([s["id"] for s in batch])
            return {"a": "你好", "b": " "} if len(batch) == 2 else {"b": "再见"}

        translate_batches(
            segments, request, lambda: checkpoints.append([s["zh"] for s in segments])
        )
        self.assertEqual(calls, [["a", "b"], ["b"]])
        self.assertEqual(checkpoints[0], ["已有译文", "你好", ""])
        self.assertEqual(segments[2]["zh"], "再见")

    def test_persistent_empty_does_not_discard_later_batches(self):
        segments = [{"id": k, "zh": ""} for k in ["a", "b", "c"]]
        calls = []

        def request(batch):
            calls.append([s["id"] for s in batch])
            return {s["id"]: "" if s["id"] == "b" else "译文" for s in batch}

        with self.assertRaisesRegex(ValueError, "b"):
            translate_batches(segments, request, batch_size=2)
        self.assertEqual(calls, [["a", "b"], ["b"], ["c"]])
        self.assertEqual([s["zh"] for s in segments], ["译文", "", "译文"])

    def test_retry_network_failure_preserves_success(self):
        segments = [{"id": "a", "zh": ""}, {"id": "b", "zh": ""}]
        calls = []

        def request(batch):
            calls.append(batch)
            if len(calls) == 2:
                raise httpx.ReadTimeout("offline test")
            return {"a": "译文", "b": ""}

        with self.assertRaises(httpx.ReadTimeout):
            translate_batches(segments, request)
        self.assertEqual(segments[0]["zh"], "译文")
        self.assertEqual(len(calls), 2)

    def test_batch_id_mismatch_not_partially_accepted(self):
        segments = [{"id": "a", "zh": ""}, {"id": "b", "zh": ""}]
        with self.assertRaises(ValueError):
            translate_batches(segments, lambda batch: {"a": "你好", "wrong": "再见"})
        self.assertEqual([s["zh"] for s in segments], ["", ""])

    def test_echo_removed_but_translation_retained(self):
        raw = '```json\n{"translations":[{"id":"a","zh":"你好","source":"こんにちは"}]}\n```'
        self.assertEqual(validate_translation(normalize_content(raw), ["a"]), {"a": "你好"})

    def test_strict_rejections_preserved(self):
        for payload in [
            {"translations": [{"id": "b", "zh": "你好", "source": "x"}]},
            {"translations": [{"id": "a", "source": "x"}]},
            {"translations": [{"id": "a", "zh": 3, "source": "x"}]},
            {
                "translations": [
                    {
                        "id": "a",
                        "zh": "你好",
                        "source": {},
                    }
                ]
            },
            {"translations": [{"id": "a", "zh": "你好", "unknown": "x", "source": "x"}]},
            {"translations": [{"id": "a", "zh": "你好"}, {"id": "a", "zh": "好"}]},
        ]:
            with self.subTest(payload=payload), self.assertRaises(TranslationError):
                validate_translation(normalize_content(json.dumps(payload)), ["a"])

    def test_malformed_not_repaired(self):
        self.assertEqual(normalize_content('{"translations":'), '{"translations":')

    def test_real_translator_contract_without_network(self):
        count = 0

        def respond(request):
            nonlocal count
            count += 1
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "translations": [
                                            {"id": "a", "zh": "你好", "source": "こんにちは"}
                                        ]
                                    }
                                )
                            },
                        }
                    ]
                },
            )

        with TranslationClient(transport=httpx.MockTransport(respond)) as client:
            with DeepSeekTranslator(
                api_key="test-only", model="test", max_retries=1, client=client
            ) as t:
                chunk = TranslationChunk(
                    sentences=[
                        Sentence(id="a", start_seconds=0, end_seconds=1, source_text="こんにちは")
                    ]
                )
                self.assertEqual(t.translate_chunk(chunk, "[]", "[]", "test"), {"a": "你好"})
        self.assertEqual(count, 1)

    def test_http_failure_preserved(self):
        with TranslationClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(401, text="denied"))
        ) as client:
            response = client.post("https://example.invalid/chat/completions")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.text, "denied")
