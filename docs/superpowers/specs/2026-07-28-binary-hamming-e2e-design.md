# Binary Hamming 端到端测试设计

## 目标

为新增的 Binary 数据类型和 Hamming 距离提供一个可重复执行的验证入口。入口先运行
与本功能相关的过滤单元测试，再使用真实的 ann-benchmarks HDF5 数据集，通过 VSAG
公开 C++ API 完成全链路验证。

测试固定使用
`/data/jingyue.zjl/ob_data/hamming/sift-256-hamming.hdf5` 作为默认数据源。
命令行允许覆盖数据路径，以便在其他环境运行。

## 文件与入口

- `examples/cpp/113_index_binary_hamming_hnsw.cpp`
  - 常驻的 C++ 端到端程序。
  - 由 `examples/cpp/CMakeLists.txt` 注册为独立可执行目标。
  - 复用仓库已有的 HDF5 1.14.4 构建产物，不增加系统依赖。
- `scripts/testing/test_binary_hamming_e2e.sh`
  - 配置并按目标构建过滤单元测试和端到端程序。
  - 只运行 `build/tests/unittests`，不执行没有匹配用例的 `functests` 和
    `tests_mockimpl`。
  - 运行 C++ 端到端程序，并把可选的第一个参数作为 HDF5 路径传入。

## 数据读取

HDF5 文件的 `/train` 和 `/test` 是形状分别为 `(988258, 256)` 和
`(1000, 256)` 的逐 bit `bool` 矩阵，VSAG 的 `BinaryVectors()` 则要求每条
256-bit 向量打包为 32 字节。

C++ 程序执行以下转换：

1. 分块读取完整的 `/train`，避免同时保留约 253 MiB 的未打包矩阵。
2. 将每连续 8 个 bool bit 打包到一个 `uint8_t`。
3. 读取并打包前 100 条 `/test`。
4. 读取对应的前 100 行 `/neighbors` 和 `/distances`。
5. 保留原始 train 行号作为 VSAG ID，使 HDF5 真值 ID 可以直接参与校验。

不能截断底库后继续使用原始 `/neighbors`：真值可能指向截断范围之外。端到端程序
因此加载全部 train，仅限制查询数量。

## 端到端流程

1. 使用 `dtype=binary`、`metric_type=hamming` 创建 HNSW。
2. 用除最后一条以外的完整底库执行 `Build()`。
3. 用最后一条底库执行 `Add()`，恢复与 HDF5 完整 train 一致的语料范围。
4. 对前 100 条查询执行 KNN Search，计算 recall@10。
5. 将首条查询的第 10 个真值距离乘以 256 作为 Range Search radius，校验所有返回
   距离均不大于 radius。
6. 使用首条查询和其真值近邻执行 `CalcDistanceById()`，同时用 C++ popcount
   计算期望 Hamming bit 数并比对。
7. 将索引流式序列化到临时文件，释放原索引后创建新索引并反序列化。
8. 重新执行查询，校验反序列化前后的 ID 和距离完全一致。

VSAG 返回的 Hamming 距离是不同 bit 的数量，范围为 `[0, 256]`；HDF5
`/distances` 保存的是归一化距离，范围为 `[0, 1]`。与 HDF5 距离比较时使用
`vsag_distance / 256`，并采用浮点容差。

## 过滤单元测试

Catch2 过滤表达式使用 `[binary],[hamming]`。现有带标签用例可以直接匹配；
包含 Binary/Hamming section、但外层缺少标签的 Factory 和 IndexCommonParam
用例将补充相同标签。脚本不使用 `--allow-running-no-tests`，因此过滤器意外匹配
零个测试时必须失败。

## 参数与失败处理

- 默认 HDF5 路径：
  `/data/jingyue.zjl/ob_data/hamming/sift-256-hamming.hdf5`
- 查询数量固定为 100，top-k 固定为 10。
- HNSW 固定使用 `max_degree=32`、`ef_construction=200` 和 `ef_search=200`。
- 平均 recall@10 低于 0.90 时返回非零。
- HDF5 结构、维度、距离类型、Build/Add/Search、距离校验或序列化任一环节失败，
  程序和 Shell 入口均立即返回非零，并输出阶段和错误原因。
- 临时索引文件使用系统临时目录，正常完成或异常退出时均清理。

## 验证策略

1. 在补测试标签前列出过滤用例，确认缺少 Factory 和 IndexCommonParam 用例。
2. 补充标签后再次列出并运行过滤用例，确认相关测试均被执行。
3. 在 C++ 目标尚未注册时先运行构建命令，确认目标缺失；随后注册并实现最小程序。
4. 使用指定 HDF5 文件运行完整端到端流程。
5. 使用不存在的 HDF5 路径运行程序，确认错误路径返回非零且信息清晰。
6. 使用 clang-format 15 格式化新增或修改的 C++ 文件，并检查所有新增文本文件
   均以换行结尾。
