"""Loader routing tests; run with python -m unittest discover -s tests."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import sys


# Keep routing tests runnable without installing the GPU training environment.
_stubs = {
    "safetensors": MagicMock(),
    "safetensors.torch": MagicMock(),
    "transformers": MagicMock(),
}
_spec = importlib.util.spec_from_file_location(
    "teacher_loading_under_test", Path(__file__).resolve().parents[1] / "teacher_loading.py"
)
loader = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, _stubs):
    _spec.loader.exec_module(loader)


class TeacherLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.weights = self.folder / "custom.safetensors"
        self.weights.touch()
        self.args = SimpleNamespace(teacher_model_path=str(self.weights))
        for name in ("AutoConfig", "AutoModelForCausalLM", "AutoTokenizer", "load_model"):
            patcher = patch.object(loader, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_single_file_loads_strictly_and_moves_after_loading(self):
        model = loader.load_teacher_model(self.args, "bf16", device="cuda")
        loader.AutoConfig.from_pretrained.assert_called_once_with(
            str(self.folder), local_files_only=True
        )
        loader.AutoModelForCausalLM.from_config.assert_called_once_with(
            loader.AutoConfig.from_pretrained.return_value, torch_dtype="bf16"
        )
        loader.load_model.assert_called_once_with(
            model, str(self.weights), strict=True, device="cpu"
        )
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with("cuda")
        loader.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_directory_uses_transformers_for_sharded_weights(self):
        self.args.teacher_model_path = str(self.folder)
        loader.load_teacher_model(self.args, "fp16")
        loader.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            str(self.folder), config=loader.AutoConfig.from_pretrained.return_value,
            torch_dtype="fp16", local_files_only=True,
        )
        loader.load_model.assert_not_called()

    def test_hub_id_remains_supported(self):
        self.args.teacher_model_path = "Qwen/Qwen3-0.6B-Base"
        loader.load_teacher_model(self.args, "bf16")
        self.assertFalse(
            loader.AutoModelForCausalLM.from_pretrained.call_args.kwargs["local_files_only"]
        )

    def test_offline_flag_applies_to_all_hub_resources(self):
        self.args.teacher_model_path = "Qwen/Qwen3-0.6B-Base"
        self.args.teacher_local_files_only = True
        loader.load_teacher_model(self.args, "bf16")
        loader.load_teacher_tokenizer(self.args)
        for method in (
            loader.AutoConfig.from_pretrained,
            loader.AutoModelForCausalLM.from_pretrained,
            loader.AutoTokenizer.from_pretrained,
        ):
            self.assertTrue(method.call_args.kwargs["local_files_only"])

    def test_tokenizer_defaults_to_weights_directory(self):
        loader.load_teacher_tokenizer(self.args, model_max_length=512)
        loader.AutoTokenizer.from_pretrained.assert_called_once_with(
            str(self.folder), local_files_only=True, model_max_length=512
        )

    def test_separate_config_and_tokenizer_sources(self):
        config_file = self.folder / "config.json"
        config_file.touch()
        self.args.teacher_config_path = str(config_file)
        self.args.teacher_tokenizer_path = "Qwen/Qwen3-0.6B-Base"
        loader.load_teacher_config(self.args)
        loader.load_teacher_tokenizer(self.args)
        loader.AutoConfig.from_pretrained.assert_called_once_with(
            str(config_file), local_files_only=True
        )
        loader.AutoTokenizer.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-0.6B-Base", local_files_only=False
        )

    def test_missing_weights_fail_before_loading_config(self):
        self.weights.unlink()
        with self.assertRaises(FileNotFoundError):
            loader.load_teacher_model(self.args, "bf16")
        loader.AutoConfig.from_pretrained.assert_not_called()

    def test_unsupported_file_fails_clearly(self):
        self.args.teacher_model_path = str(self.folder / "weights.bin")
        Path(self.args.teacher_model_path).touch()
        with self.assertRaisesRegex(ValueError, "safetensors"):
            loader.load_teacher_config(self.args)

    def test_incomplete_or_incompatible_weights_fail(self):
        loader.load_model.side_effect = RuntimeError("Missing keys")
        with self.assertRaisesRegex(RuntimeError, "checkpoint Transformers"):
            loader.load_teacher_model(self.args, "bf16")

    def test_missing_config_has_actionable_message(self):
        loader.AutoConfig.from_pretrained.side_effect = OSError("Missing config")
        with self.assertRaisesRegex(OSError, "teacher_config_path"):
            loader.load_teacher_config(self.args)

    def test_missing_tokenizer_has_actionable_message(self):
        loader.AutoTokenizer.from_pretrained.side_effect = OSError("Missing tokenizer")
        with self.assertRaisesRegex(OSError, "teacher_tokenizer_path"):
            loader.load_teacher_tokenizer(self.args)


if __name__ == "__main__":
    unittest.main()
