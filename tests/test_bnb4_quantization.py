import unittest

import torch
import torch.nn as nn

from fish_speech.models.text2semantic.llama import _convert_linear_layers_to_bnb4

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
        self.assertTrue(torch.allclose(model.proj.weight.float(), state_dict["proj.weight"]))
        self.assertTrue(
            torch.allclose(model.block[1].weight.float(), state_dict["block.1.weight"])
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
