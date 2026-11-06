import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from bson import ObjectId
from bson.binary import UUID_SUBTYPE, UuidRepresentation
from bson.codec_options import CodecOptions
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError
from pymongo.results import BulkWriteResult

from main import MODEL_NAME
from majfeed.itemcf import load_like_pairs, load_pairs, save_neighbors
from majfeed.publication import cleanup_old_versions, make_model_version, neighbor_list_id

V1 = f"{MODEL_NAME}-20261011:000001"
V2 = f"{MODEL_NAME}-20261011:000002"
OLD, NEW = UUID(int=0xA0), UUID(int=0xB0)


class MongoWriteChecks(unittest.TestCase):
    def setUp(self):
        self.uri = os.environ["MONGO_TEST_URI"]
        self.client = MongoClient(self.uri, uuidRepresentation="standard", tz_aware=True)
        # 只在本次创建的独立数据库中验证回写，结束后删除该测试库。
        self.database_name = f"majfeed_write_check_{uuid4().hex}"
        self.db = self.client[self.database_name]

    def tearDown(self):
        self.client.drop_database(self.database_name)
        self.client.close()

    def test_bson_contract_batching_and_immutable_versions(self):
        low, high = UUID(int=1), UUID("FFFFFFFF-0000-4000-8000-0000000000AB")
        sources = [UUID(int=1000 + i) for i in range(1, 502)]
        topks = {
            source: [{"chartId": low, "score": 0.25}, {"chartId": high, "score": 0.75}]
            for source in sources
        }
        written = save_neighbors(self.uri, self.database_name, topks, V1)
        self.assertEqual(written, 501)
        self.assertEqual(self.db.ChartNeighbors.count_documents({}), 501)
        docs = list(self.db.ChartNeighbors.find())
        self.assertEqual(len({doc["GeneratedAt"] for doc in docs}), 1)
        for doc in docs:
            self.assertEqual(set(doc), {"_id", "ChartId", "ModelVersion", "GeneratedAt", "Neighbors"})
            self.assertIsInstance(doc["ChartId"], UUID)
            self.assertEqual(doc["_id"], f"{V1}|{str(doc['ChartId']).lower()}")
            self.assertEqual(doc["ModelVersion"], V1)
            self.assertIsInstance(doc["GeneratedAt"], datetime)
            self.assertEqual(doc["GeneratedAt"].utcoffset(), timedelta(0))
            self.assertEqual(doc["Neighbors"], [
                {"ChartId": high, "Score": 0.75},
                {"ChartId": low, "Score": 0.25},
            ])
            self.assertIsInstance(doc["Neighbors"][0]["Score"], float)
        # mmfcapi 以 GuidRepresentation.Standard 读取：ChartId 必须是 BSON binary subtype 4，而不是字符串。
        raw = self.db.with_options(codec_options=CodecOptions(uuid_representation=UuidRepresentation.UNSPECIFIED))
        stored = raw.ChartNeighbors.find_one({"_id": neighbor_list_id(V1, sources[0])})
        self.assertEqual(stored["ChartId"].subtype, UUID_SUBTYPE)
        self.assertEqual(stored["Neighbors"][0]["ChartId"].subtype, UUID_SUBTYPE)
        self.assertEqual(stored["_id"], f"{V1}|00000000-0000-0000-0000-0000000003e9")

        self.assertEqual(save_neighbors(
            self.uri, self.database_name, {sources[0]: []}, V2
        ), 1)
        self.assertEqual(self.db.ChartNeighbors.count_documents({}), 502)
        updated = self.db.ChartNeighbors.find_one({"_id": neighbor_list_id(V2, sources[0])})
        self.assertEqual(updated["Neighbors"], [])
        self.assertEqual(self.db.ChartNeighbors.find_one({"_id": neighbor_list_id(V1, sources[0])})["ModelVersion"], V1)
        self.assertEqual(self.db.RecommendationModels.find_one({"_id": V1})["ChartCount"], 501)
        self.assertIsNotNone(self.db.RecommendationModels.find_one({"_id": V2})["PublishedAt"])
        with self.assertRaises(DuplicateKeyError):
            save_neighbors(self.uri, self.database_name, {sources[0]: []}, V1)
        self.assertEqual(self.db.ChartNeighbors.count_documents({}), 502)

    def test_empty_result_does_not_write_documents(self):
        self.assertEqual(save_neighbors(self.uri, self.database_name, {}, V1), 0)
        self.assertEqual(self.db.ChartNeighbors.count_documents({}), 0)
        self.assertEqual(self.db.RecommendationModels.count_documents({}), 0)

    def test_main_reads_configuration_and_writes_weighted_neighbors(self):
        a, b, c = [UUID(int=i) for i in range(1, 4)]
        u1, u2 = [ObjectId(f"{i:024x}") for i in range(1, 3)]
        self.db.MaiCharts.insert_many([
            {"_id": a, "Hash": "a"},
            {"_id": b, "Hash": "b"},
            {"_id": c, "Hash": "c"},
        ])
        self.db.NetUsers.insert_many([{"_id": u1}, {"_id": u2}])
        self.db.Scores.insert_many([
            {"PlayerId": u1, "Hash": "a"},
            {"PlayerId": u1, "Hash": "b"},
            {"PlayerId": u2, "Hash": "a"},
            {"PlayerId": u2, "Hash": "c"},
        ])
        # u1 点赞 A、点踩 C 在 ChartVotes；u2 点赞 B 还在旧版内嵌数组里。
        self.db.ChartVotes.insert_many([
            {"ChartHash": "a", "UserId": u1, "Kind": 1},
            {"ChartHash": "c", "UserId": u1, "Kind": -1},
        ])
        self.db.ChartInteracts.insert_many([
            {"ChartHash": "b", "Likes": [u2]},
        ])

        now = datetime.now(timezone.utc)
        for hours in (4, 3, 2):
            old_version = make_model_version(MODEL_NAME, now - timedelta(hours=hours))
            save_neighbors(self.uri, self.database_name, {OLD: []}, old_version)
            self.db.RecommendationModels.update_one(
                {"_id": old_version}, {"$set": {"PublishedAt": now - timedelta(hours=hours)}}
            )
        # 按 Hash 写入的旧模型由旧后端读取，本任务不回收它。
        legacy_version = make_model_version("itemcf", now - timedelta(hours=5))
        self.db.RecommendationModels.insert_one({
            "_id": legacy_version, "Model": "itemcf", "PublishedAt": now - timedelta(hours=5),
        })
        root = Path(__file__).resolve().parents[1]
        process = subprocess.run(
            [sys.executable, str(root / "src/main.py")],
            cwd=root,
            env={
                **os.environ,
                "MONGO_URI": self.uri,
                "MONGO_DATABASE": self.database_name,
                "TOP_K": "1",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        output = json.loads(process.stdout)
        version = output.pop("model_version")
        self.assertRegex(version, r"^itemcf-id-\d{8}:\d{6}$")
        self.assertEqual(output, {
            "k": 1,
            "play_pairs": 4,
            "like_pairs": 2,
            "written_charts": 3,
            "deleted_versions": 2,
        })

        # A 的权重为 [2.5, 1]，B 为 [1, 1.5]，两位用户分别接触 2、3 张谱面。
        expected_ab = (2.5 / math.log(3) + 1.5 / math.log(4)) / math.sqrt(7.25 * 3.25)
        docs = {doc["ChartId"]: doc for doc in self.db.ChartNeighbors.find({"ModelVersion": version})}
        self.assertEqual(set(docs), {a, b, c})
        self.assertEqual(docs[a]["_id"], neighbor_list_id(version, a))
        self.assertEqual(docs[a]["ModelVersion"], version)
        self.assertIsNotNone(self.db.RecommendationModels.find_one({"_id": version})["PublishedAt"])
        self.assertEqual(docs[a]["Neighbors"][0]["ChartId"], b)
        self.assertAlmostEqual(docs[a]["Neighbors"][0]["Score"], expected_ab, places=6)
        self.assertEqual(docs[b]["Neighbors"][0]["ChartId"], a)
        self.assertEqual(docs[c]["Neighbors"][0]["ChartId"], b)
        self.assertTrue(all(len(doc["Neighbors"]) == 1 for doc in docs.values()))
        self.assertEqual(self.db.RecommendationModels.count_documents({"Model": MODEL_NAME, "PublishedAt": {"$ne": None}}), 2)
        self.assertIsNotNone(self.db.RecommendationModels.find_one({"_id": legacy_version}))

        # 删除后按同一 Hash 重新上传会得到新 UUID：交互仍按 Hash 关联，但训练键换成新谱面的 Id；
        # 已发布的近邻仍指向旧 Id，下一轮训练才会按新 Id 生成。
        plays_before = load_pairs(self.uri, self.database_name)
        likes_before = load_like_pairs(self.uri, self.database_name)
        self.assertTrue(all(set(pair) == {"userId", "chartId", "weight"} for pair in plays_before + likes_before))
        renamed = {old: uuid4() for old in (a, b, c)}
        self.db.MaiCharts.delete_many({"_id": {"$in": [a, b, c]}})
        self.db.MaiCharts.insert_many([
            {"_id": renamed[old], "Hash": chart_hash} for old, chart_hash in zip((a, b, c), "abc")
        ])
        to_keys = lambda pairs, ids={}: {(p["userId"], ids.get(p["chartId"], p["chartId"]), p["weight"]) for p in pairs}
        self.assertEqual(to_keys(plays_before, renamed), to_keys(load_pairs(self.uri, self.database_name)))
        self.assertEqual(to_keys(likes_before, renamed), to_keys(load_like_pairs(self.uri, self.database_name)))
        self.assertEqual(self.db.ChartNeighbors.find_one({"_id": neighbor_list_id(version, a)})["Neighbors"], docs[a]["Neighbors"])
        self.assertEqual(self.db.ChartNeighbors.count_documents({"ModelVersion": version}), 3)


    def test_like_pairs_read_upvotes_and_legacy_likes_once(self):
        u1, u2, deleted = [ObjectId(f"{i:024x}") for i in range(1, 4)]
        self.db.MaiCharts.insert_many([{"_id": UUID(int=i), "Hash": h} for i, h in enumerate("abc", start=1)])
        self.db.NetUsers.insert_many([{"_id": u1}, {"_id": u2}])
        self.db.ChartVotes.insert_many([
            {"ChartHash": "a", "UserId": u1, "Kind": 1},
            {"ChartHash": "c", "UserId": u2, "Kind": 1},
            {"ChartHash": "b", "UserId": u1, "Kind": -1},  # 点踩不算点赞。
            {"ChartHash": "c", "UserId": deleted, "Kind": 1},  # 账号已删除。
            {"ChartHash": "gone", "UserId": u2, "Kind": 1},  # 谱面已删除。
        ])
        self.db.ChartInteracts.insert_many([
            {"ChartHash": "a", "Likes": [u1, u2]},  # u1 与 ChartVotes 重复，只计一次。
            {"ChartHash": "b", "Upvotes": 3},  # 已迁移的记录没有 Likes 数组。
        ])
        pairs = load_like_pairs(self.uri, self.database_name)
        self.assertEqual(
            sorted((p["userId"], p["chartId"], p["weight"]) for p in pairs),
            [(u1, UUID(int=1), 1.5), (u2, UUID(int=1), 1.5), (u2, UUID(int=3), 1.5)],
        )

    def test_partial_batch_failure_keeps_version_unpublished_and_old_data_unchanged(self):
        save_neighbors(self.uri, self.database_name, {OLD: []}, V1)
        original_write = Collection.bulk_write
        calls = 0

        def fail_second_batch(collection, operations, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated batch failure")
            return original_write(collection, operations, *args, **kwargs)

        with patch.object(Collection, "bulk_write", new=fail_second_batch):
            with self.assertRaisesRegex(RuntimeError, "simulated batch"):
                save_neighbors(self.uri, self.database_name, {UUID(int=i + 1): [] for i in range(501)}, V2)
        self.assertEqual(self.db.ChartNeighbors.count_documents({"ModelVersion": V2}), 500)
        self.assertIsNone(self.db.RecommendationModels.find_one({"_id": V2})["PublishedAt"])
        self.assertEqual(self.db.RecommendationModels.find_one(
            {"Model": MODEL_NAME, "PublishedAt": {"$ne": None}}, sort=[("_id", -1)]
        )["_id"], V1)
        self.assertIsNotNone(self.db.ChartNeighbors.find_one({"_id": neighbor_list_id(V1, OLD)}))

    def test_incomplete_write_receipt_does_not_publish_or_clean_up(self):
        save_neighbors(self.uri, self.database_name, {OLD: []}, V1)
        with patch.object(Collection, "bulk_write", return_value=BulkWriteResult({"nInserted": 0}, True)):
            with self.assertRaisesRegex(RuntimeError, "Incomplete neighbor batch"):
                save_neighbors(self.uri, self.database_name, {NEW: []}, V2)
        self.assertIsNone(self.db.RecommendationModels.find_one({"_id": V2})["PublishedAt"])
        self.assertIsNotNone(self.db.ChartNeighbors.find_one({"_id": neighbor_list_id(V1, OLD)}))

    def test_unacknowledged_write_does_not_publish(self):
        with patch.object(Collection, "bulk_write", return_value=BulkWriteResult({}, False)):
            with self.assertRaisesRegex(RuntimeError, "Incomplete neighbor batch"):
                save_neighbors(self.uri, self.database_name, {NEW: []}, V1)
        self.assertIsNone(self.db.RecommendationModels.find_one({"_id": V1})["PublishedAt"])

    def test_successful_publication_does_not_read_back_neighbor_documents(self):
        original_find = Collection.find

        def reject_neighbor_read(collection, *args, **kwargs):
            if collection.name == "ChartNeighbors":
                raise AssertionError("Publication must not read back neighbor documents")
            return original_find(collection, *args, **kwargs)

        with patch.object(Collection, "find", new=reject_neighbor_read):
            self.assertEqual(save_neighbors(self.uri, self.database_name, {NEW: []}, V1), 1)
        self.assertIsNotNone(self.db.RecommendationModels.find_one({"_id": V1})["PublishedAt"])

    def test_cleanup_preserves_two_latest_releases_and_waits_since_supersession(self):
        now = datetime.now(timezone.utc)
        versions = [f"{MODEL_NAME}-20261011:00000{i}" for i in range(1, 5)]
        ages = [timedelta(hours=3), timedelta(hours=2), timedelta(minutes=10), timedelta(minutes=5)]
        for version, age in zip(versions, ages):
            save_neighbors(self.uri, self.database_name, {OLD: []}, version)
            self.db.RecommendationModels.update_one({"_id": version}, {"$set": {"PublishedAt": now - age}})
        # Publication never deletes anything, even when older data already exceeds the grace period.
        self.assertEqual(self.db.ChartNeighbors.count_documents({}), 4)
        save_neighbors(self.uri, self.database_name, {NEW: []}, "other-20261011:000001")
        self.db.RecommendationModels.insert_one({
            "_id": f"{MODEL_NAME}-20990101:000000", "Model": MODEL_NAME, "PublishedAt": None,
        })
        self.assertEqual(cleanup_old_versions(self.db, MODEL_NAME, now=now), 1)
        self.assertIsNone(self.db.ChartNeighbors.find_one({"ModelVersion": versions[0]}))
        self.assertIsNotNone(self.db.ChartNeighbors.find_one({"ModelVersion": versions[1]}))
        self.assertEqual(cleanup_old_versions(self.db, MODEL_NAME, now=now + timedelta(hours=2)), 1)
        self.assertEqual(set(self.db.RecommendationModels.distinct("_id", {"Model": MODEL_NAME, "PublishedAt": {"$ne": None}})), set(versions[2:]))
        self.assertIsNotNone(self.db.ChartNeighbors.find_one({"ModelVersion": "other-20261011:000001"}))
        self.assertIsNotNone(self.db.RecommendationModels.find_one({"_id": f"{MODEL_NAME}-20990101:000000"}))


if __name__ == "__main__":
    unittest.main()
