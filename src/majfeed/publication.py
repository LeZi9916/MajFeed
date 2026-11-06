"""Versioned ItemCF publishing. A release is readable only after PublishedAt is set."""

from datetime import datetime, timedelta, timezone
import math
import re
from typing import TYPE_CHECKING
from uuid import UUID

from pymongo import InsertOne, MongoClient
from pymongo.write_concern import WriteConcern

if TYPE_CHECKING:
    from .itemcf import Neighbor

BATCH_SIZE = 500
RETAINED_VERSIONS = 2
CLEANUP_GRACE = timedelta(hours=1)
_MODEL_NAME = re.compile(r"[a-z][a-z0-9_-]*")


def make_model_version(model: str, generated_at: datetime | None = None) -> str:
    """UTC、固定宽度；同秒版本冲突由 Mongo 唯一 _id 拒绝，不覆盖或合并。"""
    if not _MODEL_NAME.fullmatch(model):
        raise ValueError("Model name must use lowercase letters, digits, '_' or '-'.")
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    return f"{model}-{generated_at.astimezone(timezone.utc):%Y%m%d:%H%M%S}"


def _parse_version(version: str) -> tuple[str, datetime]:
    model, _, timestamp = version.rpartition("-")
    generated_at = datetime.strptime(timestamp, "%Y%m%d:%H%M%S").replace(tzinfo=timezone.utc)
    if make_model_version(model, generated_at) != version:
        raise ValueError("Model version must be '{model}-yyyyMMdd:HHmmss' in UTC")
    return model, generated_at


def neighbor_list_id(version: str, chart_id: UUID) -> str:
    """ChartNeighbors 的 _id：`{ModelVersion}|{ChartId}`。

    ChartId 使用小写、带连字符的 36 位格式，与 .NET `Guid.ToString()` 一致，mmfcapi 按同样格式拼接查询。
    """
    return f"{version}|{chart_id}"


def _validate_neighbors(chart_id: UUID, neighbors: list["Neighbor"]) -> None:
    # 只接受 UUID 对象：旧版的 Hash 字符串或 UUID 文本都会写成错误的 BSON 类型。
    if not isinstance(chart_id, UUID):
        raise ValueError(f"Source chart id must be a UUID, got {chart_id!r}")
    seen = set()
    for neighbor in neighbors:
        target = neighbor["chartId"]
        score = float(neighbor["score"])
        if not isinstance(target, UUID) or target == chart_id or target in seen:
            raise ValueError(f"Invalid or duplicate neighbor for {chart_id}")
        if not math.isfinite(score) or score <= 0:
            raise ValueError(f"Invalid neighbor score for {chart_id}")
        seen.add(target)


def _document(chart_id: UUID, neighbors: list["Neighbor"], version: str, generated_at: datetime) -> dict:
    # ChartId 由 uuidRepresentation="standard" 写成 BSON UUID（subtype 4），与 MaiChart._id 相同。
    return {
        "_id": neighbor_list_id(version, chart_id),
        "ChartId": chart_id,
        "ModelVersion": version,
        "GeneratedAt": generated_at,
        "Neighbors": [
            {"ChartId": neighbor["chartId"], "Score": float(neighbor["score"])}
            for neighbor in sorted(neighbors, key=lambda item: (-item["score"], item["chartId"]))
        ],
    }


def cleanup_old_versions(database, model: str, *, now: datetime | None = None) -> int:
    """保留最新两版；其他版本被更高版本替代满一小时后再清理。"""
    now = now or datetime.now(timezone.utc)
    models = database["RecommendationModels"]
    neighbors = database["ChartNeighbors"]
    releases = models.find({"Model": model, "PublishedAt": {"$ne": None}}).sort("_id", -1)
    superseded_at = None
    deleted = 0
    with releases:
        for position, release in enumerate(releases):
            if position >= RETAINED_VERSIONS and superseded_at <= now - CLEANUP_GRACE:
                # 只删除明确的旧版本，不能用 != 本次版本，以免删掉并发任务发布的新版。
                neighbors.delete_many({"ModelVersion": release["_id"]})
                models.delete_one({"_id": release["_id"]})
                deleted += 1
            # 版本按生成时间排序，发布时间可能不同；首次被任一更高版本取代才开始计时。
            published_at = release["PublishedAt"]
            superseded_at = published_at if superseded_at is None else min(superseded_at, published_at)
    return deleted


def save_neighbors(uri: str, database_name: str, topks: dict[UUID, list["Neighbor"]], model_version: str) -> int:
    """占用新版本 → 分批插入并核对写入回执 → 原子发布。"""
    model, generated_at = _parse_version(model_version)
    # 空训练结果维持原有行为，不替换当前可用模型。
    if not topks:
        return 0
    # 写入前检查全部输入；避免明显的格式错误留下半批文档。
    for chart_id, neighbors in topks.items():
        _validate_neighbors(chart_id, neighbors)

    with MongoClient(uri, uuidRepresentation="standard", tz_aware=True) as client:
        database = client.get_database(database_name, write_concern=WriteConcern(w="majority"))
        collection = database["ChartNeighbors"]
        models = database["RecommendationModels"]
        collection.create_index("ModelVersion", name="ModelVersion_cleanup")
        models.create_index([("Model", 1), ("_id", -1)], name="Model_latest_version")
        # 先占位：唯一 _id 阻止同秒并发任务或重试覆盖同一版本。未发布记录不会被 API 选中。
        models.insert_one({
            "_id": model_version,
            "Model": model,
            "GeneratedAt": generated_at,
            "PublishedAt": None,
            "ChartCount": len(topks),
        })

        operations = []
        written = 0
        for chart_id, neighbors in topks.items():
            operations.append(InsertOne(_document(chart_id, neighbors, model_version, generated_at)))
            if len(operations) == BATCH_SIZE:
                result = collection.bulk_write(operations, ordered=False)
                if not result.acknowledged or result.inserted_count != len(operations):
                    raise RuntimeError("Incomplete neighbor batch")
                written += result.inserted_count
                operations.clear()
        if operations:
            result = collection.bulk_write(operations, ordered=False)
            if not result.acknowledged or result.inserted_count != len(operations):
                raise RuntimeError("Incomplete neighbor batch")
            written += result.inserted_count

        # 仅累加服务器确认的插入数；异常或短写会阻止发布，无需读回全部近邻。
        if written != len(topks):
            raise RuntimeError(f"Incomplete model: expected {len(topks)} charts, inserted {written}")
        published = models.update_one(
            {"_id": model_version, "PublishedAt": None},
            {"$set": {"PublishedAt": datetime.now(timezone.utc)}},
        )
        if published.modified_count != 1:
            raise RuntimeError("Could not publish model version")
        # 写入/校验失败时不会走到这里；保留未发布记录，重跑需使用新版本号。
    return written
