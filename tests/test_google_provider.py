import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from agent_core.llm_provider.api_registry import StreamOptions
from agent_core.llm_provider.provider_types import Model
from agent_core.llm_provider.providers.google_provider import GoogleProvider
from agent_core.llm_provider.retry import extract_retry_delay_ms


class _FakeGoogleResponse:
    def __init__(self, *, status_code, headers=None, body=b"", lines=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._lines = list(lines or [])
        self.content = body

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeGoogleStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return _FakeGoogleStreamContext(response)


class GoogleProviderTests(unittest.TestCase):
    def test_extract_retry_delay_prefers_headers(self):
        delay = extract_retry_delay_ms(
            "Please retry in 1s",
            {"Retry-After": "5"},
            now=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(delay, 6000)

    def test_extract_retry_delay_parses_reset_headers(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        reset_epoch = int(now.timestamp()) + 20

        delay = extract_retry_delay_ms(
            "",
            {"x-ratelimit-reset": str(reset_epoch)},
            now=now,
        )

        self.assertEqual(delay, 21000)

    def test_google_provider_retries_when_delay_is_within_cap(self):
        provider = GoogleProvider()
        model = Model(id="gemini-test", provider="google", api="google-gemini")
        options = StreamOptions(max_retry_delay_ms=1500)

        responses = [
            _FakeGoogleResponse(
                status_code=429,
                headers={"Retry-After": "0"},
                body=b'{"error":{"message":"Please retry in 0s"}}',
            ),
            _FakeGoogleResponse(
                status_code=200,
                lines=[
                    'data: {"candidates":[{"content":{"parts":[{"text":"hello from gemini"}]}}]}',
                    "",
                ],
            ),
        ]

        fake_client = _FakeAsyncClient(responses)

        async def collect():
            chunks = []
            with patch("agent_core.llm_provider.providers.google_provider.httpx.AsyncClient", return_value=fake_client):
                with patch("agent_core.llm_provider.providers.google_provider.asyncio.sleep") as sleep_mock:
                    async for chunk in provider.stream(model, [{"role": "user", "content": "hi"}], options, api_key="k"):
                        chunks.append(chunk)
                    sleep_mock.assert_awaited_once()
            return chunks

        chunks = asyncio.run(collect())

        self.assertEqual(chunks, [{"content": "hello from gemini"}])
        self.assertEqual(fake_client.calls, 2)

    def test_google_provider_rejects_retry_delay_above_cap(self):
        provider = GoogleProvider()
        model = Model(id="gemini-test", provider="google", api="google-gemini")
        options = StreamOptions(max_retry_delay_ms=500)

        responses = [
            _FakeGoogleResponse(
                status_code=429,
                headers={"Retry-After": "1"},
                body=b'{"error":{"message":"Please retry in 1s"}}',
            )
        ]

        fake_client = _FakeAsyncClient(responses)

        async def collect():
            with patch("agent_core.llm_provider.providers.google_provider.httpx.AsyncClient", return_value=fake_client):
                async for _chunk in provider.stream(model, [{"role": "user", "content": "hi"}], options, api_key="k"):
                    pass

        with self.assertRaisesRegex(ValueError, "max_retry_delay_ms=500"):
            asyncio.run(collect())

    def test_google_provider_emits_reasoning_for_thought_signature_parts(self):
        provider = GoogleProvider()
        model = Model(id="gemini-test", provider="google", api="google-gemini")
        options = StreamOptions()

        responses = [
            _FakeGoogleResponse(
                status_code=200,
                lines=[
                    'data: {"candidates":[{"content":{"parts":[{"text":"private plan","thoughtSignature":"sig-1"},{"text":"visible answer"}]}}]}',
                    "",
                ],
            ),
        ]

        fake_client = _FakeAsyncClient(responses)

        async def collect():
            chunks = []
            with patch("agent_core.llm_provider.providers.google_provider.httpx.AsyncClient", return_value=fake_client):
                async for chunk in provider.stream(model, [{"role": "user", "content": "hi"}], options, api_key="k"):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())

        self.assertEqual(chunks[0]["reasoning"], "private plan")
        self.assertEqual(chunks[0]["provider_state"]["gemini"]["thought_signatures"], ["sig-1"])
        self.assertEqual(chunks[1], {"content": "visible answer"})


if __name__ == "__main__":
    unittest.main()
