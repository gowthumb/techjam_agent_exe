import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import agent.llm_client as llm_client
from agent.llm_client import resolve_model, resolve_temperature


class ModelRoutingTest(unittest.TestCase):
    def test_role_specific_model_overrides_shared_model(self):
        with patch.dict(os.environ, {"CODER_MODEL": "coder"}, clear=False), patch.object(
            llm_client, "_load_configuration", return_value=("key", "url", "shared")
        ):
            self.assertEqual(resolve_model("CODER"), "coder")

    def test_role_without_override_uses_shared_model(self):
        environment = dict(os.environ)
        environment.pop("DEBUGGER_MODEL", None)
        with patch.dict(os.environ, environment, clear=True), patch.object(
            llm_client, "_load_configuration", return_value=("key", "url", "shared")
        ):
            self.assertEqual(resolve_model("DEBUGGER"), "shared")

    def test_role_temperature_defaults_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=False):
            for role in ("CODER", "DEBUGGER", "PLANNER"):
                os.environ.pop(role + "_TEMPERATURE", None)
            self.assertEqual(resolve_temperature("CODER"), 0.15)
            self.assertEqual(resolve_temperature("DEBUGGER"), 0.15)
            self.assertEqual(resolve_temperature("PLANNER"), 0.6)
            self.assertIsNone(resolve_temperature("OTHER"))
        with patch.dict(os.environ, {"CODER_TEMPERATURE": "0.25"}, clear=False):
            self.assertEqual(resolve_temperature("CODER"), 0.25)

    def test_reset_quota_pause_budget_starts_a_new_run_cleanly(self):
        llm_client._QUOTA_PAUSE_COUNT = 3
        llm_client.reset_quota_pause_budget()
        self.assertEqual(llm_client._QUOTA_PAUSE_COUNT, 0)

    def test_request_lifecycle_logs_metadata_without_contents(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="secret response"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: completion)))
        with patch.object(llm_client, "OpenAI", return_value=client), patch.object(
            llm_client, "_load_configuration", return_value=("secret-key", "https://example.invalid/v1", "test-model")
        ), patch("builtins.print") as print_mock:
            response = llm_client.call_llm("system", "user", role="CODER")
        output = "\n".join(" ".join(str(value) for value in call.args) for call in print_mock.call_args_list)
        self.assertEqual((response.input_tokens, response.output_tokens), (3, 2))
        self.assertIn("role=CODER", output)
        self.assertIn("model=test-model", output)
        self.assertIn("prompt_chars=10", output)
        self.assertIn("elapsed_s=", output)
        self.assertNotIn("secret response", output)
        self.assertNotIn("secret-key", output)

    def test_call_omits_temperature_when_unspecified(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        create = unittest.mock.Mock(return_value=completion)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm_client, "OpenAI", return_value=client), patch.object(
            llm_client, "_load_configuration", return_value=("key", "url", "model")
        ), patch("builtins.print"):
            llm_client.call_llm("system", "user")
        self.assertNotIn("temperature", create.call_args.kwargs)