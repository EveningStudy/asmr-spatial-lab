"""Normalize one harmless model echo before the upstream strict validator.

Only a string-valued `source` inside translations is removed. Unknown fields,
malformed JSON, IDs, ordering and translated text remain the validator's job.
"""

import json
import re

import httpx


def translate_batches(segments, request_batch, checkpoint=None, batch_size=20):
    """Keep valid results immediately; retry only empty items once, individually.

    Transport/schema/refusal errors are not retried here. Nothing is silently
    dropped or replaced by the Japanese source. The caller persists on failure.
    """
    pending = [s for s in segments if not s.get("zh", "").strip()]
    unresolved = []

    def accept(batch, mapping):
        expected = [s["id"] for s in batch]
        if not isinstance(mapping, dict) or list(mapping) != expected:
            raise ValueError("翻译返回 ID 或顺序不一致；本次响应未采纳。")
        if any(not isinstance(text, str) for text in mapping.values()):
            raise ValueError("翻译返回的译文不是字符串；本次响应未采纳。")
        empty = []
        for segment in batch:
            text = mapping[segment["id"]].strip()
            if text:
                segment["zh"] = text
            else:
                empty.append(segment)
        if checkpoint is not None:
            checkpoint()
        return empty

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        empty = accept(batch, request_batch(batch))
        for segment in empty:
            print(
                f"翻译 {segment['id']} 返回空值；其他有效译文已保存，仅单独重试此句一次。",
                flush=True,
            )
            if accept([segment], request_batch([segment])):
                unresolved.append(segment["id"])
    if unresolved:
        raise ValueError(
            "已保存其余有效译文，以下句子单独重试后仍为空："
            + ", ".join(unresolved)
            + "。请检查 transcript.json 对应日文并手动填写 zh；未跳过台词，也未开始配音。"
        )


def normalize_content(content: str) -> str:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except (ValueError, TypeError):
        return content
    if not isinstance(payload, dict) or not isinstance(payload.get("translations"), list):
        return content
    changed = False
    for item in payload["translations"]:
        if isinstance(item, dict) and isinstance(item.get("source"), str):
            del item["source"]
            changed = True
    return json.dumps(payload, ensure_ascii=False) if changed else content


class TranslationClient(httpx.Client):
    def post(self, url, **kwargs):
        response = super().post(url, **kwargs)
        if not response.is_success:
            return response
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            original = message["content"]
            if not isinstance(original, str):
                return response
            normalized = normalize_content(original)
            if normalized == original:
                return response
            message["content"] = normalized
        except (ValueError, KeyError, IndexError, TypeError):
            return response
        # Use a new buffered response; do not monkey-patch upstream global validation.
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
        }
        replacement = httpx.Response(
            response.status_code, json=data, headers=headers, request=response.request
        )
        response.close()
        return replacement
