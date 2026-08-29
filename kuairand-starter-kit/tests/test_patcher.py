import unittest

from agent.patcher import SearchMatchError, apply_patch, validate_syntax


class PatcherSmokeTest(unittest.TestCase):
    def test_replaces_a_uniquely_matching_search_block(self):
        diff = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
        self.assertEqual(apply_patch("value = 1\n", diff), "value = 2\n")

    def test_reports_non_unique_searches(self):
        diff = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
        with self.assertRaisesRegex(SearchMatchError, "matched 2 times"):
            apply_patch("value = 1\nvalue = 1\n", diff)

    def test_rejects_invalid_python(self):
        with self.assertRaisesRegex(SyntaxError, "invalid Python syntax"):
            validate_syntax("def broken(:\n")


if __name__ == "__main__":
    unittest.main()