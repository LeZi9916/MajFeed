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


PLAY_HISTORY_COLLECTION = "PlayHistory"
SCORE_COLLECTION = "Scores"


def _play_history_stages() -> list[dict]:
    """游玩记录已按谱面 Id 记录：关联现存谱面，$unwind 会排除已删除谱面的游玩。"""
    return [
        {
            "$lookup": {
                "from": "MaiCharts",
                "localField": "ChartId",
                "foreignField": "_id",
                "pipeline": [{"$project": {"_id": 1}}],
                "as": "chart",
            }
        },
        {"$unwind": "$chart"},
        {"$project": {"userId": "$PlayerId", "chartId": "$chart._id"}},
    ]


def _score_stages() -> list[dict]:
    """成绩只记 level hash：每个 level hash 归到它的代表谱面，没有现存谱面的 level hash 被排除。"""
    return [
        # 每个 level hash 只查一次代表谱面，再展开回各个玩家。
        {"$group": {"_id": "$LevelHash", "players": {"$addToSet": "$PlayerId"}}},
        {
            "$lookup": {
                "from": "MaiChartMetadata",
                "localField": "_id",
                "foreignField": "SHA256",
                "pipeline": [
                    {
                        "$lookup": {
                            "from": "MaiCharts",
                            "localField": "ChartId",
                            "foreignField": "_id",
                            "pipeline": [{"$project": {"Timestamp": 1}}],
                            "as": "chart",
                        }
                    },
                    {"$unwind": "$chart"},
                    # 与 mmfcapi 的 MaiScoreSubmissionService.OrderByRepresentative 相同：
                    # 最早上传的现存谱面，同时上传时按谱面 Id，再按难度序号。
                    {"$sort": {"chart.Timestamp": 1, "ChartId": 1, "Level": 1}},
                    {"$limit": 1},
                ],
                "as": "level",
            }
        },
        {"$unwind": "$level"},
        {"$unwind": "$players"},
        {"$project": {"userId": "$players", "chartId": "$level.ChartId"}},
    ]


def load_pairs(
    uri: str,
    database_name: str,
    *,
    collection: str = SCORE_COLLECTION,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[InteractionPair]:
    """提取 [start, end) 内去重后的用户—谱面对，游玩权重为 1.0。

    collection 为 Scores（每位玩家每个 level hash 一条最佳成绩）或 PlayHistory（每次游玩一条）。
    """
    if collection == SCORE_COLLECTION:
        chart_stages = _score_stages()
    elif collection == PLAY_HISTORY_COLLECTION:
        chart_stages = _play_history_stages()
    else:
        raise ValueError(f"Unsupported interaction collection: {collection}")

    # 先按时间筛选再去重，分别保留训练期和验证期实际出现的交互。
    time_filter = {}
    if start is not None:
        time_filter["$gte"] = start
    if end is not None:
        time_filter["$lt"] = end
    pipeline = [{"$match": {"Timestamp": time_filter}}] if time_filter else []
    pipeline.extend(chart_stages)
    pipeline.extend([
        # 忽略难度差异，同一用户和同一谱面只保留一条游玩关系。
        {"$group": {"_id": {"userId": "$userId", "chartId": "$chartId"}}},
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
UPVOTE = 1  # mmfcapi 的 VoteKind.Upvote，枚举按整数存储；点踩为 -1。


def load_like_pairs(uri: str, database_name: str) -> list[InteractionPair]:
    """提取当前有效的点赞关系，点赞权重为 1.5。"""
    # 点赞存放在 ChartVotes，每位用户对每张谱面一条文档，按谱面 Id 记录；只取 upvote，点踩不参与训练。
    # UpdatedAt 是最后一次改票的时间，不是点赞发生的时间，因此不用于历史时间切分。
    pipeline = [
        {"$match": {"Kind": UPVOTE}},
        # 只保留现存谱面；迁移时找不到谱面、仍只有 ChartHash 的投票在这里被排除。
        {
            "$lookup": {
                "from": "MaiCharts",
                "localField": "ChartId",
                "foreignField": "_id",
                "pipeline": [{"$project": {"_id": 1}}],
                "as": "chart",
            }
        },
        {"$unwind": "$chart"},
        # 只保留仍然存在的账号，避免将已删除账号的点赞加入训练。
        {
            "$lookup": {
                "from": "NetUsers",
                "localField": "UserId",
                "foreignField": "_id",
                "pipeline": [{"$project": {"_id": 1}}],
                "as": "user",
            }
        },
        {"$unwind": "$user"},
        # ItemCF 会累加重复坐标，重复的点赞必须在这里去掉，否则权重变成 3.0。
        {"$group": {"_id": {"userId": "$UserId", "chartId": "$chart._id"}}},
        {
            "$project": {
                "_id": 0,
                "userId": "$_id.userId",
                "chartId": "$_id.chartId",
                "weight": {"$literal": 1.5},
            }
        },
    ]
    with MongoClient(uri, uuidRepresentation="standard", tz_aware=True) as client:
        with client[database_name][VOTE_COLLECTION].aggregate(pipeline) as cursor:
            return list(cursor)


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
