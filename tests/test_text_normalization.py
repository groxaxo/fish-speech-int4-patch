import unittest

from fish_speech.text import TextNormalizationOptions, normalize_text_for_tts


class TextNormalizationTests(unittest.TestCase):
    def test_normalizes_email_addresses(self):
        text = normalize_text_for_tts("Contact user@example.com for support.")
        self.assertIn("user at example dot com", text)

    def test_normalizes_urls(self):
        text = normalize_text_for_tts("Visit https://example.com/path now.")
        self.assertIn("https example dot com slash path", text)

    def test_normalizes_money(self):
        text = normalize_text_for_tts("The price is $50.30.")
        self.assertIn("fifty dollars and thirty cents", text)

    def test_normalizes_phone_numbers(self):
        text = normalize_text_for_tts("Call (555) 123-4567 today.")
        self.assertIn("five five five", text)
        self.assertIn("one two three", text)

    def test_bypass_normalization_when_disabled(self):
        text = normalize_text_for_tts(
            "Visit https://example.com",
            normalize=False,
            normalization_options=TextNormalizationOptions(),
        )
        self.assertEqual(text, "Visit https://example.com")


if __name__ == "__main__":
    unittest.main()
