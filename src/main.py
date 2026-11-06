import json
import os
from itertools import chain

from pymongo import MongoClient
from pymongo.write_concern import WriteConcern

from majfeed.itemcf import build_itemcf, load_like_pairs, load_pairs, save_neighbors
from majfeed.publication import cleanup_old_versions, make_model_version

# 近邻按 ChartId 写入。与按 Hash 写入的旧模型 itemcf 互不可读，因此换用新模型名：
# 旧后端继续读 itemcf 的最后一版，切到 ChartId 的后端读 itemcf-id，切换期间没有空窗。
# 本任务只回收 itemcf-id 的旧版本，itemcf 的数据在后端切换稳定后手动删除。
MODEL_NAME = "itemcf-id"


def run(uri: str, database_name: str, model_name: str = MODEL_NAME, k: int = 100) -> dict:
    """读取游玩与点赞，计算 Top-K，并回写当前谱面的近邻列表。"""
    model_version = make_model_version(model_name)
    play_pairs = load_pairs(uri, database_name)
    like_pairs = load_like_pairs(uri, database_name)
    # 保留两种行为的权重，同一用户—谱面对在 ItemCF 内累计为 2.5。
    topks = build_itemcf(chain(play_pairs, like_pairs), k=k)
    written = save_neighbors(uri, database_name, topks, model_version)
    deleted = 0
    # 只有本轮确实发布成功才清理；同一个夜间任务完成训练、发布和旧版本回收。
    if written:
        with MongoClient(uri, uuidRepresentation="standard", tz_aware=True) as client:
            database = client.get_database(database_name, write_concern=WriteConcern(w="majority"))
            deleted = cleanup_old_versions(database, model_name)
    return {
        "model_version": model_version,
        "k": k,
        "play_pairs": len(play_pairs),
        "like_pairs": len(like_pairs),
        "written_charts": written,
        "deleted_versions": deleted,
    }


def main():
    # 本地由 uv --env-file 加载配置，容器中由 Docker 注入同名环境变量。
    result = run(
        os.environ["MONGO_URI"],
        os.environ["MONGO_DATABASE"],
        MODEL_NAME,
        k=int(os.environ.get("TOP_K", "100")),
    )
    # 只在整轮写入完成后输出统计，异常直接交给调用脚本或定时任务处理。
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
