"""State reducer tests; stdlib-only so these run in the lightweight CI job."""
import unittest

from core.state import merge_dict


class TestMergeDict(unittest.TestCase):
    def test_parallel_llm_totals_recomputed_from_merged_stage_payloads(self):
        common = {
            "extract": {"total_tokens": 50},
            "llm_total_tokens": 50,
        }
        geology = {
            **common,
            "geology": {"total_tokens": 30},
            "llm_total_tokens": 80,
        }
        wellbore = {
            **common,
            "wellbore": {"total_tokens": 20},
            "llm_total_tokens": 70,
        }

        merged = merge_dict(geology, wellbore)

        self.assertEqual(merged["extract"]["total_tokens"], 50)
        self.assertEqual(merged["geology"]["total_tokens"], 30)
        self.assertEqual(merged["wellbore"]["total_tokens"], 20)
        self.assertEqual(merged["llm_total_tokens"], 100)

    def test_merge_preserves_non_token_routing_payloads(self):
        merged = merge_dict({"extract": "local:qwen2.5"}, {"grade": "local:qwen2.5"})
        self.assertEqual(merged, {"extract": "local:qwen2.5", "grade": "local:qwen2.5"})


if __name__ == "__main__":
    unittest.main()
