import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID

from majfeed.publication import _document, make_model_version, neighbor_list_id, save_neighbors

A = UUID("0F1E2D3C-4B5A-6978-8796-A5B4C3D2E1F0")
B, C, D = [UUID(int=i) for i in range(2, 5)]


class PublicationTests(unittest.TestCase):
    def test_version_uses_fixed_width_utc_and_sorts_across_midnight(self):
        local = datetime(2026, 10, 12, 8, 0, 1, tzinfo=timezone(timedelta(hours=8)))
        self.assertEqual(make_model_version("itemcf", local), "itemcf-20261012:000001")
        self.assertLess("itemcf-20261011:235959", make_model_version("itemcf", local))

    def test_invalid_version_or_neighbors_fail_before_opening_database(self):
        with patch("majfeed.publication.MongoClient") as client:
            for version in ("manual-v1", "itemcf-20261012:1", "other|bad-20261012:000001"):
                with self.subTest(version=version), self.assertRaises(ValueError):
                    save_neighbors("unused", "unused", {A: []}, version)
            for neighbors in ([{"chartId": A, "score": 1}], [{"chartId": B, "score": float("nan")}],
                              [{"chartId": B, "score": 0}], [{"chartId": B, "score": 1}] * 2,
                              [{"chartId": str(B), "score": 1}]):
                with self.subTest(neighbors=neighbors), self.assertRaises(ValueError):
                    save_neighbors("unused", "unused", {A: neighbors}, "itemcf-id-20261012:000001")
            # 旧版的 Hash 字符串和 UUID 文本都不能作为源谱面写入。
            for source in ("hash-a", str(A)):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    save_neighbors("unused", "unused", {source: []}, "itemcf-id-20261012:000001")
            client.assert_not_called()

    def test_document_keys_neighbors_by_chart_id(self):
        version = "itemcf-id-20261012:000001"
        generated_at = datetime(2026, 10, 12, tzinfo=timezone.utc)
        neighbors = [{"chartId": C, "score": 0.5}, {"chartId": B, "score": 0.5}, {"chartId": D, "score": 0.9}]
        doc = _document(A, neighbors, version, generated_at)
        # _id 的 ChartId 部分与 .NET Guid.ToString() 一致：小写、带连字符。
        self.assertEqual(doc["_id"], "itemcf-id-20261012:000001|0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
        self.assertEqual(doc["_id"], neighbor_list_id(version, A))
        self.assertEqual(set(doc), {"_id", "ChartId", "ModelVersion", "GeneratedAt", "Neighbors"})
        self.assertEqual(doc["ChartId"], A)
        # 分数降序，同分按 ChartId 升序。
        self.assertEqual(doc["Neighbors"], [
            {"ChartId": D, "Score": 0.9},
            {"ChartId": B, "Score": 0.5},
            {"ChartId": C, "Score": 0.5},
        ])

    def test_version_accepts_the_chart_id_model_name(self):
        generated_at = datetime(2026, 10, 12, 0, 0, 1, tzinfo=timezone.utc)
        self.assertEqual(make_model_version("itemcf-id", generated_at), "itemcf-id-20261012:000001")

    def test_version_requires_a_valid_model_name_and_timezone(self):
        with self.assertRaises(ValueError):
            make_model_version("itemcf|invalid")
        with self.assertRaises(ValueError):
            make_model_version("itemcf-id", datetime(2026, 10, 12))


if __name__ == "__main__":
    unittest.main()
