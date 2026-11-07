import math
import unittest
from uuid import UUID

from bson import ObjectId

from majfeed.itemcf import build_itemcf, load_pairs


class ItemCFTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d = [UUID(int=i) for i in range(1, 5)]
        # User activity: 3, 2, 1, 1. Item support: 3, 2, 1, 1.
        histories = [(self.a, self.b, self.c), (self.a, self.b), (self.a,), (self.d,)]
        self.pairs = [
            {"userId": ObjectId(f"{u:024x}"), "chartId": chart, "weight": 1.0}
            for u, history in enumerate(histories, start=1)
            for chart in history
        ]

    def test_iuf_scores_and_neighbor_contract(self):
        neighbors = build_itemcf(self.pairs)
        expected_ab = (1 / math.log(4) + 1 / math.log(3)) / math.sqrt(6)
        expected_ac = (1 / math.log(4)) / math.sqrt(3)
        self.assertEqual([n["chartId"] for n in neighbors[self.a]], [self.b, self.c])
        self.assertAlmostEqual(neighbors[self.a][0]["score"], expected_ab, places=6)
        self.assertAlmostEqual(neighbors[self.a][1]["score"], expected_ac, places=6)
        self.assertEqual(neighbors[self.d], [])
        for chart, items in neighbors.items():
            for neighbor in items:
                self.assertEqual(set(neighbor), {"chartId", "score"})
                self.assertIsInstance(neighbor["chartId"], UUID)
                self.assertIsInstance(neighbor["score"], float)
                self.assertNotEqual(chart, neighbor["chartId"])

    def test_top_k_limits_each_chart(self):
        neighbors = build_itemcf(self.pairs, k=1)
        self.assertEqual([n["chartId"] for n in neighbors[self.a]], [self.b])
        self.assertTrue(all(len(items) <= 1 for items in neighbors.values()))

    def test_like_weight_uses_distinct_activity_and_weighted_column_norms(self):
        user = ObjectId(f"{1:024x}")
        pairs = self.pairs + [{"userId": user, "chartId": self.a, "weight": 1.5}]
        neighbors = build_itemcf(pairs)

        # User 1 still has three charts; A's weights are now [2.5, 1, 1].
        expected_ab = (2.5 / math.log(4) + 1 / math.log(3)) / math.sqrt(8.25 * 2)
        expected_ac = (2.5 / math.log(4)) / math.sqrt(8.25)
        scores = {n["chartId"]: n["score"] for n in neighbors[self.a]}
        self.assertAlmostEqual(scores[self.b], expected_ab, places=6)
        self.assertAlmostEqual(scores[self.c], expected_ac, places=6)

    def test_repeated_pairs_match_preaggregated_weights(self):
        user = ObjectId(f"{1:024x}")
        repeated = self.pairs + [{"userId": user, "chartId": self.a, "weight": 1.5}]
        aggregated = [
            {**pair, "weight": 2.5}
            if pair["userId"] == user and pair["chartId"] == self.a
            else pair
            for pair in self.pairs
        ]
        self.assertEqual(build_itemcf(repeated), build_itemcf(aggregated))

    def test_load_pairs_rejects_other_collections_before_connecting(self):
        # 只有 Scores 和 PlayHistory 的文档形状已知；其他集合在连接数据库之前就被拒绝。
        with self.assertRaisesRegex(ValueError, "ChartVotes"):
            load_pairs("mongodb://127.0.0.1:1", "unused", collection="ChartVotes")

    def test_keys_are_the_chart_ids_of_the_pairs(self):
        user = ObjectId(f"{1:024x}")
        ids = [UUID("0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"), UUID("ffffffff-0000-4000-8000-000000000001")]
        neighbors = build_itemcf({"userId": user, "chartId": chart, "weight": 1.0} for chart in ids)
        self.assertEqual(set(neighbors), set(ids))
        self.assertEqual(neighbors[ids[0]][0]["chartId"], ids[1])


if __name__ == "__main__":
    unittest.main()
