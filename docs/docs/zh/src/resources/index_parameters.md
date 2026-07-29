# 索引参数

本页汇总 VSAG 各索引类型的常用参数。完整枚举请参考源码：

- 构建参数键：`src/constants.cpp`
- 公开常量：`include/vsag/constants.h`
- 每个索引的示例：`examples/cpp/*_index_*.cpp`（例如 `103_index_hgraph.cpp`）

## 通用参数

所有索引在构建时都需要提供以下顶层字段：

| 字段 | 取值 | 说明 |
|------|------|------|
| `dim` | 正整数 | 向量维度，构建后不可更改 |
| `dtype` | `float32` / `fp16` / `bf16` / `int8` / `binary` | 向量数据类型；`binary` 当前仅支持 HNSW |
| `metric_type` | `l2` / `ip` / `cosine` / `hamming` / `l1` | 距离度量；`hamming` 必须配合 `dtype: binary`；`l1` 仅支持 `float32` 普通 HNSW |

使用 `dtype: "binary"` 时，`dim` 表示逻辑 bit 数，必须大于 0 且能被 8 整除。每条向量以
`dim / 8` 个压缩字节传入。

## HNSW

普通 HNSW 支持 packed binary 向量和 Hamming 距离：

```json
{
    "dim": 128,
    "dtype": "binary",
    "metric_type": "hamming",
    "hnsw": {
        "max_degree": 16,
        "ef_construction": 100
    }
}
```

binary 路径支持构建、追加、搜索、更新、序列化和原始向量读取；HNSW Merge、Pretrain 与
量化路径暂不支持 binary。其他索引类型会拒绝 `dtype: "binary"`。

普通 HNSW 还支持 `float32` 向量的 L1 距离：

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

L1 距离为各维绝对差之和，不执行向量归一化。该路径使用 generic 实现，不包含 SIMD
优化，并保持普通 float32 HNSW 的构建、追加、搜索、更新、距离计算、序列化、Merge 和
Pretrain 能力。`fresh_hnsw` 及其他索引类型不接受 `metric_type: "l1"`。

## HGraph

HGraph 的构建参数使用通用的 `index_param` 键（参见 `examples/cpp/103_index_hgraph.cpp`）；
`hgraph` 键则保留给搜索期参数。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 400
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `max_degree` | 16~48 | 每节点最大出边数 |
| `ef_construction` | 200~500 | 构建阶段候选集大小，越大召回越高、构建越慢 |
| `base_quantization_type` | `fp32` / `fp16` / `bf16` / `sq8` / `sq4` / `pq` | 主存储的量化策略 —— 支持的全部取值见[量化章节](../quantization/README.md) |

搜索时：

```json
{"hgraph": {"ef_search": 100}}
```

`hgraph` 搜索参数还接受 `brute_force_threshold`（`[0.0, 1.0]` 区间的 float，
默认 `0.0`）。当取值 `> 0` 且当前请求的 filter 的 `ValidRatio()` 不超过该
阈值时，HGraph 会跳过图遍历，直接在通过过滤的 id 上做精确暴扫。详见
[HGraph 索引文档](../indexes/hgraph.md#高选择性过滤下的暴搜回退brute_force_threshold)。

## LazyHGraph

LazyHGraph 的构建参数可以放在顶层 `lazy_hgraph` 对象中（推荐，语义更清晰），也可以放在
通用的 `index_param` 对象中。`hgraph` 子对象会转交给转换后的内部 HGraph。

```json
{
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "lazy_hgraph": {
        "transition_threshold": 1000,
        "hgraph": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
}
```

| 字段 | 典型值 | 说明 |
|------|-------|------|
| `transition_threshold` | `1000` 或按业务规模设置 | 从精确 flat 搜索转换到 HGraph 的正整数向量数量阈值 |
| `hgraph` | HGraph 构建对象 | graph 阶段的参数；见 [HGraph](../indexes/hgraph.md) |

LazyHGraph 只支持 `dtype: "float32"`。搜索参数使用 `hgraph` 对象，例如
`{"hgraph": {"ef_search": 100}}`。详见 [LazyHGraph 索引文档](../indexes/lazy_hgraph.md)。

`hgraph` 搜索参数还接受以下 filter 相关参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_ratio` | float | `0.2` | 控制带 filter 搜索时跳过候选检查的比例，取值范围为 `[0.0, 1.0]`。值越大，跳过越激进，搜索越快但可能影响召回。 |
| `skip_strategy` | string | `"deterministic_accumulative"` | 跳过策略。支持 `"random"` 和 `"deterministic_accumulative"`。 |

## IVF

```json
{
    "ivf": {
        "nlist": 4096,
        "base_quantization_type": "sq8",
        "nprobe": 32
    }
}
```

## Brute Force

```json
{"brute_force": {}}
```

无需额外参数。

## Pyramid

Pyramid 支持按 tag 组织多棵子图：

```json
{
    "pyramid": {
        "tag_dim": 1,
        "max_degree": 24,
        "ef_construction": 300
    }
}
```

## SINDI（稀疏向量）

```json
{
    "sindi": {
        "top_k": 32,
        "doc_prune_ratio": 0.1
    }
}
```

## 运行期参数

除构建参数外，`Index::Tune` 与 `SearchParam` 可在运行时调整 `ef_search`、`nprobe` 等参数。参考
[优化器](../advanced/optimizer.md) 与各 `examples/cpp/3xx_feature_*.cpp` 示例。
