import unittest

from tools.run_webui import parse_args as parse_webui_args


class CliDefaultTests(unittest.TestCase):
    def test_webui_defaults_to_bnb4_and_fp16(self):
        args = parse_webui_args([])
        self.assertTrue(args.bnb4)
        self.assertTrue(args.half)
        self.assertEqual(args.theme, "system")


if __name__ == "__main__":
    unittest.main()
