## Summary

为MajdataNet设计的推荐推流数据处理部分，并持久化到MongoDB。Feed的拼接和构成API由MajdataNet后端负责。

## Notes

推荐算法：`ItemCF + 用户活跃度IUF降权`，同时权重部分采用户游玩记录为1.0，点赞记录为1.5，在合并时加总。IUF依然靠`log1p(uniquePlayCount)`作为降权。

分母类似普通ItemCF，不过由于我们引入权重，这里靠`sigma sqrt(R^2)`作为Normalization范数。公式：

\[
S_{ij}
=
\frac{\sum_u IUF_u\,R_{ui}R_{uj}}
{\sqrt{\sum_u R_{ui}^{2}}\sqrt{\sum_u R_{uj}^{2}}}
\]

使用了`sparse_dot_topn`库在计算过程中就截断topn，减少一下在服务器上的内存占用

## 主入口

### 开发环境
在项目根目录准备 `.env` 后运行：

```bash
cp .env.example .env
uv run --locked --env-file .env python src/main.py
```

### 远程Docker

main直接负责所有步骤，一次性完成退出。

- 本轮完整发布成功后执行旧版本回收。
- 保留版本号最大的两版。
- 更旧版本自首次被更高版本取代满一小时后，才删除其近邻及发布记录。

```bash
docker run --rm --platform linux/amd64 \
  -e MONGO_URI='mongodb://<user>:<password>@<host>:<port>/?authSource=admin' \
  -e MONGO_DATABASE='<db name>' \
  sniperpigeon/majfeed:<version>
```

`MONGO_URI` 的 `<host>:<port>` 是**从容器内部**看到的地址，容器里的 `127.0.0.1` 指向容器自己：

- Mongo 在同一个 Docker 网络（如 compose 服务名 `mongo`）：加 `--network <网络名>`，写 `mongo:27017`。
- Mongo 跑在宿主机或映射到宿主机端口：写 `host.docker.internal:<映射端口>`；Linux 上需要再加 `--add-host=host.docker.internal:host-gateway`。
- 密码里有 `@ : / ? #` 等字符时需 URL 编码（如 `@` → `%40`）；用户建在业务库而非 `admin` 时，`authSource` 改为对应库名。

`MONGO_URI`、`MONGO_DATABASE` 必填，`TOP_K` 默认 100，不需要动。
成功后输出 JSON，包含本轮自动生成的 `model_version`、训练交互数、写入谱面数及 `deleted_versions`。
首次上线时先跑一次 majfeed，确认 `RecommendationModels` 里有 `PublishedAt` 非空的版本，再部署 mmfcapi。


## 谱面标识与版本更新

训练、评估和近邻回写统一使用 `MaiCharts.Hash`，区分大小写。游玩从
`Scores.Hash` / `PlayHistory.Hash` 读取；点赞从 `ChartVotes` 中 `Kind = 1`（upvote）的
`ChartHash` / `UserId` 读取，点踩不参与训练。旧版后端内嵌在 `ChartInteracts.Likes` 的点赞
也会读取，后端启动迁移删掉该数组后自然不再命中，因此新旧后端都能训练。
仅保留能关联到当前谱面、且账号仍存在的交互，并按用户与 Hash 去重。
输入为 `userId/chartHash/weight`，Top-K 返回 `dict[str, list[{"hash": str, "score": float}]]`。

每轮开始自动生成 `{model}-yyyyMMdd:HHmmss`（UTC），例如 `itemcf-20261012:000001`作为可排序的版本号，方便快速定位新版。

`ChartNeighbors` 文档包含 `_id = ModelVersion + "|" + ChartHash`、
`ChartHash`、`ModelVersion`、UTC `GeneratedAt` 和 `Neighbors[{ChartHash, Score}]`。
同 Hash 更换 UUID 不影响关联。新旧版本同时存在，已发布版本不可修改。

更新数据在 `src/majfeed/publication.py`：

1. 在 `RecommendationModels` 插入版本占位：`_id`、`Model`、`GeneratedAt`、`ChartCount` 和空 `PublishedAt`。
2. 写入前检查 Hash、分数及重复项；每批 500 条插入近邻，要求 `majority` 确认，检查每批 `acknowledged` 和 `inserted_count`，最终核对累计插入数。无需全量回读或额外统计查询。
3. 全部写入确认及数量核对成功后原子设置 `PublishedAt`。mmfcapi 只在当前模型已发布记录中排序选最新版，一次推荐固定一个版本。

写入失败的情况下不会被读取，依然使用旧版本。
清理由 `src/main.py` 的 `run` 在本轮发布成功后执行，发布失败或空结果不会触发清理。


## 测试

```bash
PYTHONPATH=src uv run --locked python -m unittest discover -s tests -v
```

MongoDB 发布/失败恢复/夜间清理集成检查（只创建和清理独立测试库）：

```bash
MONGO_TEST_URI='mongodb://127.0.0.1:27018' PYTHONPATH=src \
  uv run --locked python tests/check_mongo_write.py -v
```
