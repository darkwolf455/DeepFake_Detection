"""
Focused tests for CLIPDeepfakeDetector failure-mode changes.

1. A failing CLIPVisionModelWithProjection.from_pretrained() must raise
   RuntimeError (not swallow the error or fall back to CLIPModel).

2. A missing PEFT library when use_lora=True must raise ImportError
   (not silently freeze the backbone and continue).

Does not download model weights or run training.

Run with:
    python -m unittest verisense/tests/test_model_failure_modes.py
"""

import sys
import types
import unittest
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image.model import CLIPDeepfakeDetector


class TestBackboneLoadFailure(unittest.TestCase):
    """CLIPVisionModelWithProjection load failure must raise RuntimeError."""

    def _make_failing_pretrained(self, original_exc):
        """Return a mock from_pretrained that always raises *original_exc*."""
        def _fail(*args, **kwargs):
            raise original_exc
        return _fail

    def test_os_error_becomes_runtime_error(self):
        """A network / filesystem error is wrapped in RuntimeError."""
        original = OSError("no route to host")
        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            side_effect=original,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                CLIPDeepfakeDetector(model_name="openai/clip-vit-large-patch14")

        msg = str(ctx.exception)
        self.assertIn("openai/clip-vit-large-patch14", msg)
        self.assertIn("CLIPVisionModelWithProjection", msg)

    def test_original_exception_is_chained(self):
        """The RuntimeError must chain (__cause__) the original exception."""
        original = OSError("weights missing")
        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            side_effect=original,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                CLIPDeepfakeDetector(model_name="openai/clip-vit-large-patch14")

        self.assertIs(ctx.exception.__cause__, original)

    def test_value_error_from_pretrained_becomes_runtime_error(self):
        """Any exception type from from_pretrained is wrapped, not propagated raw."""
        original = ValueError("unexpected config key")
        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            side_effect=original,
        ):
            with self.assertRaises(RuntimeError):
                CLIPDeepfakeDetector(model_name="openai/clip-vit-large-patch14")

    def test_no_clipmodel_fallback(self):
        """CLIPModel must never be imported or called as a fallback."""
        original = OSError("pretend hub is down")
        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            side_effect=original,
        ):
            # Patch CLIPModel at the transformers level to detect any call
            with mock.patch("transformers.CLIPModel") as clip_model_mock:
                with self.assertRaises(RuntimeError):
                    CLIPDeepfakeDetector(model_name="openai/clip-vit-large-patch14")
                clip_model_mock.from_pretrained.assert_not_called()


class TestPEFTImportFailure(unittest.TestCase):
    """Missing PEFT when use_lora=True must raise ImportError, not continue silently."""

    def _make_dummy_backbone(self):
        """Minimal fake backbone with a .config that satisfies __init__ without real weights."""
        cfg = types.SimpleNamespace(projection_dim=768)
        backbone = mock.MagicMock()
        backbone.config = cfg
        backbone.parameters.return_value = iter([])
        return backbone

    def test_missing_peft_raises_import_error(self):
        """ImportError is raised immediately when peft is not importable."""
        dummy_backbone = self._make_dummy_backbone()

        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            return_value=dummy_backbone,
        ):
            # Simulate peft not being installed by making the import fail
            with mock.patch.dict(sys.modules, {"peft": None}):
                with self.assertRaises(ImportError) as ctx:
                    CLIPDeepfakeDetector(
                        model_name="openai/clip-vit-large-patch14",
                        use_lora=True,
                    )

        msg = str(ctx.exception)
        self.assertIn("peft", msg.lower())
        self.assertIn("pip install peft", msg)

    def test_missing_peft_does_not_silently_continue(self):
        """After PEFT ImportError, no model object is returned — the call raises."""
        dummy_backbone = self._make_dummy_backbone()

        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            return_value=dummy_backbone,
        ):
            with mock.patch.dict(sys.modules, {"peft": None}):
                result = None
                try:
                    result = CLIPDeepfakeDetector(
                        model_name="openai/clip-vit-large-patch14",
                        use_lora=True,
                    )
                except ImportError:
                    pass

        self.assertIsNone(result, "Constructor must not return a model when PEFT is unavailable")

    def test_peft_error_is_not_runtime_error(self):
        """The raised exception is specifically ImportError, not RuntimeError or ValueError."""
        dummy_backbone = self._make_dummy_backbone()

        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            return_value=dummy_backbone,
        ):
            with mock.patch.dict(sys.modules, {"peft": None}):
                with self.assertRaises(ImportError):
                    CLIPDeepfakeDetector(
                        model_name="openai/clip-vit-large-patch14",
                        use_lora=True,
                    )
                # Confirm it is NOT a RuntimeError
                try:
                    CLIPDeepfakeDetector(
                        model_name="openai/clip-vit-large-patch14",
                        use_lora=True,
                    )
                except RuntimeError:
                    self.fail("Should raise ImportError, not RuntimeError")
                except ImportError:
                    pass  # expected

    def test_use_lora_false_does_not_require_peft(self):
        """use_lora=False must succeed even when peft is absent."""
        dummy_backbone = self._make_dummy_backbone()

        with mock.patch(
            "src.image.model.CLIPVisionModelWithProjection.from_pretrained",
            return_value=dummy_backbone,
        ):
            with mock.patch.dict(sys.modules, {"peft": None}):
                # Must not raise
                model = CLIPDeepfakeDetector(
                    model_name="openai/clip-vit-large-patch14",
                    use_lora=False,
                )
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
