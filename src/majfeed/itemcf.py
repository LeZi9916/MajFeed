from collections.abc import Iterable
from datetime import datetime
from typing import TypedDict
from uuid import UUID

import numpy as np
from bson import ObjectId
from pymongo import MongoClient
from scipy.sparse import coo_matrix
from sparse_dot_topn import sp_matmul_topn
from .publication import save_neighbors


class InteractionPair(TypedDict):
    userId: ObjectId
    chartId: UUID  # MaiChart.Id
    weight: float


class Neighbor(TypedDict):
    chartId: UUID
    score: float


def load_pairs(
    uri: str,
    database_name: str,
    *,
    collection: str = "Scores",
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[InteractionPair]:
    """提取 [start, end) 内去重后的用户—谱面对，游玩权重为 1.0。"""
    pipeline = []
    # 先按时间筛选再去重，分别保留训练期和验证期实际出现的交互。
    time_filter = {}
    if start is not None:
        time_filter["$gte"] = start
    if end is not None:
        time_filter["$lt"] = end
    if time_filter:
        pipeline.append({"$match": {"Timestamp": time_filter}})

    # 成绩仍以 maidata MD5 为键：按 Hash 关联当前谱面并取其 Id，$unwind 会排除找不到谱面的成绩记录。
    # 上传时拒绝重复 Hash，一个 Hash 只对应一张现存谱面。
    pipeline.extend([
        {
            "$lookup": {
                "from": "MaiCharts",
                "localField": "Hash",
                "foreignField": "Hash",
                "as": "chart",
            }
        },
        {"$unwind": "$chart"},
        # 忽略难度差异，同一用户和同一谱面只保留一条游玩关系。
        {
            "$group": {
                "_id": {
                    "userId": "$PlayerId",
                    "chartId": "$chart._id",
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "userId": "$_id.userId",
                "chartId": "$_id.chartId",
                "weight": {"$literal": 1.0},  # 用 $literal 返回常量，而不是保留同名字段。
            }
        },
    ])

    with MongoClient(uri, uuidRepresentation="standard", tz_aware=True) as client:
        with client[database_name][collection].aggregate(pipeline) as cursor:
            return list(cursor)


VOTE_COLLECTION = "ChartVotes"
LEGACY_LIKE_COLLECTION = "ChartInteracts"
UPVOTE = 1  # mmfcapi 的 VoteKind.Upvote，枚举按整数存储；点踩为 -1。


def _like_pipeline(match: dict, user_field: str, expand: list[dict] | None = None) -> list[dict]:
    """筛选点赞 → 按 Hash 关联当前谱面 → 排除已删除账号 → 按用户与谱面 Id 去重。"""
    return [
        {"$match": match},
        {
            "$lookup": {
                "from": "MaiCharts",
                "localField": "ChartHash",
                "foreignField": "Hash",
                "as": "chart",
            }
        },
        {"$unwind": "$chart"},
        *(expand or []),
        # 只保留仍然存在的账号，避免将已删除账号的点赞加入训练。
        {
            "$lookup": {
                "from": "NetUsers",
                "localField": user_field,
                "foreignField": "_id",
                "as": "user",
            }
        },
        {"$unwind": "$user"},
        {
            "$group": {
                "_id": {
                    "userId": f"${user_field}",
                    "chartId": "$chart._id",
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "userId": "$_id.userId",
                "chartId": "$_id.chartId",
                "weight": {"$literal": 1.5},
            }
        },
    ]


def load_like_pairs(uri: str, database_name: str) -> list[InteractionPair]:
    """提取当前有效的点赞关系，点赞权重为 1.5。"""
    # 点赞存放在 ChartVotes，每位用户对每张谱面一条文档；只取 upvote，点踩不参与训练。
    # UpdatedAt 是最后一次改票的时间，不是点赞发生的时间，因此不用于历史时间切分。
    votes = _like_pipeline({"Kind": UPVOTE}, "UserId")
    # 旧版后端把点赞内嵌在 ChartInteracts.Likes；新版后端启动时迁入 ChartVotes 并删除该数组。
    # 两处都读，后端升级前后都不会丢点赞；迁移中途两边都有的同一对只保留一条。
    legacy = _like_pipeline(
        {"Likes.0": {"$exists": True}},
        "Likes",
        [{"$unwind": "$Likes"}],  # 将点赞用户数组展开成单个用户—谱面对。
    )

    pairs: dict[tuple, InteractionPair] = {}
    with MongoClient(uri, uuidRepresentation="standard", tz_aware=True) as client:
        database = client[database_name]
        for collection, pipeline in ((VOTE_COLLECTION, votes), (LEGACY_LIKE_COLLECTION, legacy)):
            with database[collection].aggregate(pipeline) as cursor:
                for pair in cursor:
                    # ItemCF 会累加重复坐标，重复的点赞必须在这里去掉，否则权重变成 3.0。
                    pairs.setdefault((pair["userId"], pair["chartId"]), pair)
    return list(pairs.values())


def build_itemcf(
    pairs: Iterable[InteractionPair], k: int = 100
) -> dict[UUID, list[Neighbor]]:
    """计算带交互权重的 IUF ItemCF，按谱面 Id 返回 Top-K 的 chartId/score 列表。"""
    # 谱面 UUID / 用户 ObjectId 映射到连续整数索引：用户对应行，谱面对应列。
    user_to_idx = {}
    chart_to_idx = {}
    rows = []
    cols = []
    vals = []
    for record in pairs:
        user_id = record["userId"]
        chart_id = record["chartId"]
        if user_id not in user_to_idx:
            user_to_idx[user_id] = len(user_to_idx)
        if chart_id not in chart_to_idx:
            chart_to_idx[chart_id] = len(chart_to_idx)
        rows.append(user_to_idx[user_id])
        cols.append(chart_to_idx[chart_id])
        vals.append(record["weight"])

    # COO 转 CSR 时会合并重复坐标，游玩 1.0 加点赞 1.5 就得到权重 2.5。
    vals = np.asarray(vals, dtype=np.float32)
    play_record_matrix = coo_matrix( #sparse dot topn 需要 csr_matrix
        (vals, (rows, cols)), shape=(len(user_to_idx), len(chart_to_idx))
    ).tocsr()
    idx_to_chart = {v: key for key, v in chart_to_idx.items()}

    # IUF 按用户接触过的不同谱面数量计算
    user_counts = play_record_matrix.count_nonzero(axis=1).astype(np.float32)
    # 分母使用原始权重列的 L2 范数；全为 1 时，等价于谱面用户数的平方根。
    item_square_sums = (
        np.asarray(play_record_matrix.multiply(play_record_matrix).sum(axis=0)).ravel() # 换matrix后的求和
    )
    item_norm_vector = 1 / np.sqrt(item_square_sums)
    # 转为 (用户数, 1) 的列向量，让每个用户的 IUF 权重广播到其整行。
    user_iuf = (1.0 / np.log1p(user_counts))[:, None].astype(np.float32)
    # 两侧先乘 sqrt(IUF)，转置相乘后，共同用户的贡献恰好乘一次 IUF。
    
    # 换用 sparse_dot_topn 计算 Top-K，的主部分
    # 为了topk正常，先把norm也乘进去
    weighted_matrix = play_record_matrix.multiply(np.sqrt(user_iuf)) \
        .multiply(item_norm_vector).astype(np.float32).tocsr()
    similarity_matrix = sp_matmul_topn(weighted_matrix.T.tocsr(), weighted_matrix, top_n=k+1, sort=True).tocsr() #还有自身在里面，k+1

    similarity_matrix.setdiag(0)  # 清除自身相似度
    similarity_matrix.eliminate_zeros()  # 删除 setdiag 留下的显式零。

    # TODO 这里全量返回在优化后变成了最吃内存的，可以直接丢回稀疏矩阵让上层处理分块组装写入。
    topks = {}
    for i in range(similarity_matrix.shape[0]): # type:ignore
        # indptr 给出该行在 indices/data 两个一维数组中的起止位置。
        start = similarity_matrix.indptr[i]
        end = similarity_matrix.indptr[i + 1]
        neighbor_indices = similarity_matrix.indices[start:end]
        neighbor_scores = similarity_matrix.data[start:end]
        topks[idx_to_chart[i]] = [
            {
                "chartId": idx_to_chart[int(neighbor_indices[idx])],
                "score": float(neighbor_scores[idx]),
            }
            for idx in range(min(k,len(neighbor_indices)))
        ]
    return topks
