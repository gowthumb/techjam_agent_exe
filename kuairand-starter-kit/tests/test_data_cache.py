import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import agent.data_cache as data_cache


class DataCacheSmokeTest(unittest.TestCase):
    def test_fields_change_invalidates_encoded_cache(self):
        fields = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
        fields_with_extra = fields + ["hour_bucket"]
        splits = {name: [("row",)] for name in ("train", "valid", "test")}

        def fake_encode(_splits):
            field_count = len(data_cache.data.FIELDS)
            encoded = {
                name: (
                    np.zeros((1, field_count), dtype=np.int32),
                    np.ones(1, dtype=np.float32),
                    ["user"],
                )
                for name in splits
            }
            return encoded, field_count

        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(data_cache, "load_splits", return_value=splits), patch.object(
                data_cache.data, "encode", side_effect=fake_encode
            ) as encode:
                with patch.object(data_cache.data, "FIELDS", fields):
                    data_cache.load_and_encode("unused-data-dir", temporary_directory)
                    data_cache.load_and_encode("unused-data-dir", temporary_directory)

                with patch.object(data_cache.data, "FIELDS", fields_with_extra):
                    data_cache.load_and_encode("unused-data-dir", temporary_directory)

        self.assertEqual(encode.call_count, 2)


if __name__ == "__main__":
    unittest.main()