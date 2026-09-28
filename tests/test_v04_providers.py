"""Tests for the v0.4 provider layer: HTTP endpoint target, parameter fallback, SDK shape handling."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from autoredteam.providers.base import ProviderConfigurationError, ProviderRateLimitError, ProviderRequestError, TargetSpec


class _ChatbotHandler(BaseHTTPRequestHandler):
    """A tiny app API: {"query": ..., "history": [...]} -> {"data": {"answer": ...}}."""

    requests: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).requests.append({"body": body, "headers": dict(self.headers)})
        if body.get("query") == "trigger-429":
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b"slow down")
            return
        answer = f"echo:{body.get('query')}:turns={len(body.get('history', []))}"
        payload = json.dumps({"data": {"answer": answer}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestHTTPEndpointProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _ChatbotHandler)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/chat"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _ChatbotHandler.requests = []

    def _session(self, **meta):
        from autoredteam.providers.registry import get_provider_registry

        defaults = {
            "http_body": '{"query": "{{prompt}}", "history": {{history}}, "sid": "{{session_id}}"}',
            "http_response_path": "data.answer",
            "http_headers": {"Authorization": "Bearer {{env.ART_TEST_TOKEN}}"},
        }
        defaults.update(meta)
        spec = TargetSpec(provider="http", model="", endpoint=self.url, metadata=defaults)
        return get_provider_registry().create_session(spec)

    def test_templating_multi_turn_and_env_headers(self):
        with mock.patch.dict(os.environ, {"ART_TEST_TOKEN": "s3cr3t"}):
            session = self._session()
            first = session.send_user_turn("hi \"there\"")
            second = session.send_user_turn("again")
        self.assertEqual(first.text, 'echo:hi "there":turns=1')
        self.assertEqual(second.text, "echo:again:turns=3")
        sent = _ChatbotHandler.requests[1]
        self.assertEqual(sent["headers"]["Authorization"], "Bearer s3cr3t")
        self.assertIsInstance(sent["body"]["history"], list)
        self.assertEqual(sent["body"]["sid"], _ChatbotHandler.requests[0]["body"]["sid"])

    def test_reset_rotates_session_and_clears_history(self):
        with mock.patch.dict(os.environ, {"ART_TEST_TOKEN": "x"}):
            session = self._session()
            session.send_user_turn("a")
            session.reset()
            reply = session.send_user_turn("b")
        self.assertEqual(reply.text, "echo:b:turns=1")
        self.assertNotEqual(_ChatbotHandler.requests[0]["body"]["sid"], _ChatbotHandler.requests[1]["body"]["sid"])

    def test_missing_env_var_is_a_configuration_error(self):
        os.environ.pop("ART_TEST_TOKEN", None)
        with self.assertRaises(ProviderConfigurationError):
            self._session().send_user_turn("hi")

    def test_429_maps_to_rate_limit(self):
        with mock.patch.dict(os.environ, {"ART_TEST_TOKEN": "x"}):
            with self.assertRaises(ProviderRateLimitError):
                self._session().send_user_turn("trigger-429")

    def test_cli_run_against_http_endpoint_and_redacts_literal_secrets(self):
        from autoredteam.cli import main

        tmp = tempfile.mkdtemp(prefix="art-http-")
        with redirect_stdout(io.StringIO()):
            code = main([
                "run", "--provider", "http", "--endpoint", self.url, "--quiet",
                "--max-probes", "3", "--output-dir", tmp,
                "--http-body", '{"query": "{{prompt}}"}',
                "--http-response-path", "data.answer",
                "--http-header", "X-Api-Key: literal-secret",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(len(_ChatbotHandler.requests), 3)
        manifest = (Path(tmp) / "campaign_manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("literal-secret", manifest)
        self.assertIn("***redacted***", manifest)


class TestHTTPTemplateHelpers(unittest.TestCase):
    def test_placeholder_inside_string_is_not_requoted(self):
        from autoredteam.providers.http_endpoint import render_body

        body, ctype = render_body('{"q": "Hello {{prompt}}!", "m": {{messages}}}', {"prompt": "x", "messages": [1]})
        self.assertEqual(ctype, "application/json")
        self.assertEqual(json.loads(body), {"q": "Hello x!", "m": [1]})

    def test_autodetects_common_response_shapes(self):
        from autoredteam.providers.http_endpoint import extract_text

        self.assertEqual(extract_text({"choices": [{"message": {"content": "oa"}}]}), "oa")
        self.assertEqual(extract_text({"content": [{"type": "text", "text": "an"}]}), "an")
        self.assertEqual(extract_text({"response": "plain"}), "plain")
        self.assertEqual(extract_text("raw text"), "raw text")


class TestParamFallback(unittest.TestCase):
    def test_drops_unsupported_temperature_and_remembers(self):
        from autoredteam.providers._compat import ParamMemo

        calls = []

        def fake_create(**kwargs):
            calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return "ok"

        memo = ParamMemo()
        self.assertEqual(memo.call(fake_create, {"model": "m", "temperature": 0.0}), "ok")
        self.assertEqual(memo.call(fake_create, {"model": "m", "temperature": 0.0}), "ok")
        self.assertEqual(len(calls), 3)
        self.assertNotIn("temperature", calls[-1])

    def test_renames_max_tokens_for_openai(self):
        from autoredteam.providers._compat import ParamMemo

        def fake_create(**kwargs):
            if "max_tokens" in kwargs:
                raise RuntimeError("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
            return kwargs

        self.assertEqual(ParamMemo().call(fake_create, {"max_tokens": 5}), {"max_completion_tokens": 5})

    def test_other_errors_are_classified(self):
        from autoredteam.providers._compat import call_with_param_fallback

        class RateLimitError(Exception):
            status_code = 429

        with self.assertRaises(ProviderRateLimitError):
            call_with_param_fallback(lambda **_: (_ for _ in ()).throw(RateLimitError("slow")), {})
        with self.assertRaises(ProviderRequestError):
            call_with_param_fallback(lambda **_: (_ for _ in ()).throw(ValueError("boom")), {})


class TestSDKShapes(unittest.TestCase):
    def test_anthropic_skips_thinking_blocks(self):
        from autoredteam.providers.anthropic_direct import text_from_content_blocks

        blocks = [SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text="answer")]
        self.assertEqual(text_from_content_blocks(blocks), "answer")

    def test_openai_session_uses_responses_api_and_keeps_history(self):
        from autoredteam.providers.openai_direct import OpenAIDirectSession

        import sys

        fake_client = mock.MagicMock()
        fake_client.responses.create.return_value = SimpleNamespace(output_text="hello", status="completed", id="resp_1")
        fake_openai = SimpleNamespace(OpenAI=mock.MagicMock(return_value=fake_client))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), \
                mock.patch.dict(sys.modules, {"openai": fake_openai}):
            session = OpenAIDirectSession(TargetSpec(provider="openai", model="gpt-5.6-luna", system_prompt="sys"))
            session.send_user_turn("one")
            session.send_user_turn("two")
        kwargs = fake_client.responses.create.call_args.kwargs
        self.assertEqual(kwargs["instructions"], "sys")
        self.assertEqual([m["content"] for m in kwargs["input"]], ["one", "hello", "two"])
        self.assertIn("max_output_tokens", kwargs)

    def test_catalog_resolves_current_aliases(self):
        from autoredteam.providers.catalog import resolve_model_id

        self.assertEqual(resolve_model_id("anthropic", "claude-haiku-4-5"), "claude-haiku-4-5-20251001")
        self.assertEqual(resolve_model_id("bedrock", "claude-opus-5-5"), "anthropic.claude-opus-5-5")
        self.assertEqual(resolve_model_id("bedrock", "global.anthropic.claude-sonnet-5"), "global.anthropic.claude-sonnet-5")


if __name__ == "__main__":
    unittest.main()
