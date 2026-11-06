import unittest
from uuid import UUID

from bson import ObjectId

from majfeed.evaluate import evaluate_recall

'''## 时间切分召回评估

评估使用 `PlayHistory.Timestamp`：训练集为 `< split`，验证集为
`[split, end)`。先筛选时间再对用户—谱面对去重，不使用 `Scores` 的最后更新时间
代替真实游玩时间。IUF 和相似度只用训练集计算。
当前点赞没有历史时间戳，不加入该评估；每条去重游玩 pair 的权重仍为 1.0。

```bash
PYTHONPATH=src uv run --locked python -m majfeed.evaluate \
  --split '2026-09-01T00:00:00+08:00' \
  --end '2026-10-01T00:00:00+08:00' \
  --neighbor-k 100 --k 20
```

时间参数使用带时区的 ISO 8601 格式。默认读取本机
`mongodb://127.0.0.1:27018/MajNetDb_local`，可用 `--mongo-uri` 和 `--database` 修改。

对每个有训练历史的验证用户，累加其历史谱面邻居的相似度，排除已经玩过的谱面，
按分数取前 K 项；同分时按谱面 Id 排序。没有候选命中时记零，不补热门或随机结果。
验证目标为该用户在验证期首次接触、且训练期其他用户已经接触过的谱面。
`recall_at_k` 是每个可评估用户的 `命中数 / 目标数` 的算术平均。

输出还包含训练/验证规模、可评估用户数、命中数，以及被排除的冷用户、
冷用户交互、老用户的新谱交互和重复游玩数量。没有有效验证目标的用户不参与平均。
这是基于当前 `MaiCharts` 的历史回测：成绩按 Hash 关联到当前谱面并换成其 Id，
已删除或变更 Hash 且无法关联的谱面不在结果中。
'''


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d = [UUID(int=i) for i in range(1, 5)]

    def pairs(self, *entries):
        return [
            {"userId": ObjectId(f"{user:024x}"), "chartId": chart, "weight": 1.0}
            for user, chart in entries
        ]

    def test_macro_recall_and_excluded_cohorts(self):
        a, b, c, d = self.a, self.b, self.c, self.d
        train = self.pairs((1, a), (2, b), (3, c))
        validation = self.pairs((1, a), (1, b), (1, c), (1, d), (4, b), (2, c), (3, c))
        topks = {
            a: [{"chartId": b, "score": 0.9}, {"chartId": c, "score": 0.1}],
            b: [{"chartId": c, "score": 0.9}],
            c: [],
        }
        result = evaluate_recall(train, validation, topks, k=1)
        # User 1 recalls 1/2 and user 2 recalls 1/1: macro recall is 0.75.
        self.assertEqual(result["recall_at_k"], 0.75)
        self.assertEqual(result["evaluated_users"], 2)
        self.assertEqual(result["target_pairs"], 3)
        self.assertEqual(result["hits"], 2)
        self.assertEqual(result["excluded_repeat_pairs"], 2)
        self.assertEqual(result["excluded_cold_users"], 1)
        self.assertEqual(result["excluded_cold_user_pairs"], 1)
        self.assertEqual(result["excluded_cold_item_pairs"], 1)

    def test_combines_history_scores_and_filters_seen_items(self):
        a, b, c, d = self.a, self.b, self.c, self.d
        train = self.pairs((1, a), (1, b), (2, c), (3, d))
        validation = self.pairs((1, c))
        topks = {
            a: [
                {"chartId": b, "score": 0.9},
                {"chartId": d, "score": 0.6},
                {"chartId": c, "score": 0.4},
            ],
            b: [{"chartId": c, "score": 0.4}],
            c: [],
            d: [],
        }
        self.assertEqual(evaluate_recall(train, validation, topks, k=1)["recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
