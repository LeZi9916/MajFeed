import argparse
import json
from collections import defaultdict
from datetime import datetime
from statistics import mean

from .itemcf import InteractionPair, Neighbor, build_itemcf, load_pairs


def evaluate_recall(
    train_pairs: list[InteractionPair],
    validation_pairs: list[InteractionPair],
    topks: dict[str, list[Neighbor]],
    k: int = 20,
) -> dict:
    """评估老用户发现训练期已出现谱面的用户平均 Recall@K。"""
    train_by_user = defaultdict(set)
    validation_by_user = defaultdict(set)
    for pair in train_pairs:
        train_by_user[pair["userId"]].add(pair["chartHash"])
    for pair in validation_pairs:
        validation_by_user[pair["userId"]].add(pair["chartHash"])
    train_items = {pair["chartHash"] for pair in train_pairs}

    recalls = []
    hits = target_pairs = repeat_pairs = cold_users = cold_user_pairs = cold_item_pairs = 0
    for user, validation_items in validation_by_user.items():
        # 没有训练历史的用户属于冷启动，单独计数，不加入本轮个性化指标。
        if user not in train_by_user:
            cold_users += 1
            cold_user_pairs += len(validation_items)
            continue

        # 目标是用户没玩过、但训练期其他用户玩过的谱面；复玩和新谱单独计数。
        history = train_by_user[user]
        repeat_pairs += len(validation_items & history)
        novel_items = validation_items - history
        cold_item_pairs += len(novel_items - train_items)
        targets = novel_items & train_items
        if not targets:
            continue

        # 汇总历史谱面带来的邻居分数，同一候选被多条历史召回时分数累加。
        scores = defaultdict(float)
        for chart in sorted(history):
            for neighbor in topks[chart]:
                if neighbor["hash"] not in history:
                    scores[neighbor["hash"]] += neighbor["score"]
        ranked = sorted(scores, key=lambda chart: (-scores[chart], chart))[:k]
        user_hits = len(set(ranked) & targets)
        # 先算每个用户的召回率，再求平均；不要用总命中数除以总目标数代替。
        recalls.append(user_hits / len(targets))
        hits += user_hits
        target_pairs += len(targets)

    return {
        "k": k,
        "recall_at_k": mean(recalls),
        "train_pairs": len(train_pairs),
        "train_users": len(train_by_user),
        "train_items": len(train_items),
        "validation_pairs": len(validation_pairs),
        "validation_users": len(validation_by_user),
        "evaluated_users": len(recalls),
        "target_pairs": target_pairs,
        "hits": hits,
        "excluded_repeat_pairs": repeat_pairs,
        "excluded_cold_users": cold_users,
        "excluded_cold_user_pairs": cold_user_pairs,
        "excluded_cold_item_pairs": cold_item_pairs,
    }


def run_time_split(
    uri: str,
    database_name: str,
    split: datetime,
    end: datetime,
    neighbor_k: int = 100,
    k: int = 20,
) -> dict:
    """split 前训练，[split, end) 验证，使用实际游玩时间划分数据。"""
    # Scores 记录最后更新时间，回测应读 PlayHistory；点赞只有最后改票时间，不加入回测。
    train_pairs = load_pairs(uri, database_name, collection="PlayHistory", end=split)
    validation_pairs = load_pairs(
        uri, database_name, collection="PlayHistory", start=split, end=end
    )
    # neighbor_k 是每张谱面保留的邻居数，k 是最终给每个用户推荐的数量。
    topks = build_itemcf(train_pairs, k=neighbor_k)
    return {
        "split": split.isoformat(),
        "validation_end": end.isoformat(),
        "neighbor_k": neighbor_k,
        **evaluate_recall(train_pairs, validation_pairs, topks, k=k),
    }


def main():
    parser = argparse.ArgumentParser(description="Time-split ItemCF Recall@K evaluation")
    parser.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27018")
    parser.add_argument("--database", default="MajNetDb_local")
    parser.add_argument("--split", type=datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--neighbor-k", type=int, default=100)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    result = run_time_split(
        args.mongo_uri, args.database, args.split, args.end, args.neighbor_k, args.k
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
