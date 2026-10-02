import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.download_hotpotqa import export_from_huggingface, hf_row_to_hotpot_sample


class HotpotDownloadFallbackTests(unittest.TestCase):
    def test_hf_row_to_hotpot_sample_shape(self) -> None:
        row = {
            "id": "abc",
            "question": "Q?",
            "answer": "A",
            "type": "comparison",
            "level": "hard",
            "context": {
                "title": ["T1", "T2"],
                "sentences": [["s1a", "s1b"], ["s2a"]],
            },
            "supporting_facts": {"title": ["T1"], "sent_id": [1]},
        }
        sample = hf_row_to_hotpot_sample(row)
        self.assertEqual(sample["_id"], "abc")
        self.assertEqual(sample["context"], [["T1", ["s1a", "s1b"]], ["T2", ["s2a"]]])
        self.assertEqual(sample["supporting_facts"], [["T1", 1]])

    def test_export_from_huggingface_writes_official_json(self) -> None:
        fake_rows = [
            {
                "id": "1",
                "question": "q",
                "answer": "a",
                "type": "bridge",
                "level": "easy",
                "context": {"title": ["T"], "sentences": [["hello"]]},
                "supporting_facts": {"title": ["T"], "sent_id": [0]},
            }
        ]
        fake_ds = MagicMock()
        fake_ds.__iter__.return_value = iter(fake_rows)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hotpot.json"
            with patch("datasets.load_dataset", return_value=fake_ds) as mocked:
                path = export_from_huggingface(output_path=out, force=True)
            mocked.assert_called_once()
            payload = path.read_text(encoding="utf-8")
            self.assertIn('"_id": "1"', payload)
            self.assertIn('"supporting_facts": [["T", 0]]', payload)


if __name__ == "__main__":
    unittest.main()
