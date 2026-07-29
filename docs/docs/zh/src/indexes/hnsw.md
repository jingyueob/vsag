# HNSW

HNSW 是基于分层近邻图的内存索引，支持构建、追加、KNN 搜索、范围搜索、更新、距离计算和
序列化等操作。

## L1 距离

普通 HNSW 支持 `float32` 数据的 L1 距离。构建参数示例：

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l1",
    "hnsw": {
        "max_degree": 16,
        "ef_construction": 100
    }
}
```

两个向量的 L1 距离定义为各维绝对差之和：

```text
distance(a, b) = sum(abs(a[i] - b[i]))
```

L1 路径不会归一化向量，只提供 generic 实现，不包含 SIMD 优化。它支持普通 float32
HNSW 的 `Build`、`Add`、`KnnSearch`、`RangeSearch`、`UpdateId`、`UpdateVector`、
`CalcDistanceById`、`Serialize` / `Deserialize`、`Merge` 和 `Pretrain`。

当前限制：

- 仅支持 `dtype: "float32"`。
- 仅支持普通 `hnsw`，不支持 `fresh_hnsw` 或其他索引。
- 配置值必须写为小写 `"l1"`，不接受 `"manhattan"` 或 `"L1"`。

搜索参数与其他普通 HNSW 距离一致：

```json
{
    "hnsw": {
        "ef_search": 100
    }
}
```
