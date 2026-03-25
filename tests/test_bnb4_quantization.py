import unittest

import torch
import torch.nn as nn

from fish_speech.models.text2semantic.llama import (
    _convert_linear_layers_to_bnb4,
    _load_prequantized_bnb4_state_dict,
)

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


@unittest.skipUnless(bnb is not None, "bitsandbytes is not installed")
class Bnb4QuantizationTests(unittest.TestCase):
    def test_convert_replaces_nested_linear_layers(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)
                self.proj = nn.Linear(4, 4)
                self.block = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2, bias=False))

        model = TinyModel()
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        _convert_linear_layers_to_bnb4(model, compute_dtype=torch.float16)

        self.assertIsInstance(model.embedding, nn.Embedding)
        self.assertIsInstance(model.proj, bnb.nn.Linear4bit)
        self.assertIsInstance(model.block[1], bnb.nn.Linear4bit)
        self.assertTrue(
            torch.allclose(model.proj.weight.float(), state_dict["proj.weight"])
        )
        self.assertTrue(
            torch.allclose(model.block[1].weight.float(), state_dict["block.1.weight"])
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for bnb4 export")
    def test_prequantized_state_dict_can_be_reloaded(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)
                self.proj = nn.Linear(4, 4)
                self.block = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2, bias=False))

            def forward(self, x):
                x = self.embedding(x)
                x = self.proj(x)
                return self.block(x)

        original = TinyModel()
        source_state = {k: v.clone() for k, v in original.state_dict().items()}

        exported = TinyModel()
        _convert_linear_layers_to_bnb4(exported, compute_dtype=torch.float16)
        exported.load_state_dict(source_state, strict=False)
        exported = exported.to(dtype=torch.float16)
        exported = exported.to(device="cuda")
        quantized_state = exported.state_dict()

        reloaded = TinyModel()
        _convert_linear_layers_to_bnb4(reloaded, compute_dtype=torch.float16)
        remaining_state = quantized_state.copy()
        consumed_keys = _load_prequantized_bnb4_state_dict(reloaded, remaining_state)
        missing, unexpected = reloaded.load_state_dict(remaining_state, strict=False)
        missing = [key for key in missing if key not in consumed_keys]

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertIsInstance(reloaded.proj.weight, bnb.nn.Params4bit)

        reloaded = reloaded.to(device="cuda")
        out = reloaded(torch.randint(0, 8, (1, 2), device="cuda"))
        self.assertEqual(tuple(out.shape), (1, 2, 2))
        self.assertTrue(reloaded.proj.weight.bnb_quantized)


if __name__ == "__main__":
    unittest.main()
