# SGLang HiCache + Mooncake（1~3 结构梳理）

本文聚焦 `mooncake_store.py` 在 SGLang HiCache 中的角色，按 1）到 3）整理：

1) 启动 Mooncake 的主链路（谁创建了 `MooncakeStore`）  
2) `mooncake_store.py` 的初始化逻辑  
3) 运行期主要数据流向（L3<->L2）

---

## 1) 启动 Mooncake 的主链路（谁创建了 `MooncakeStore`）

### 1.1 参数入口

典型启动参数：

- `--enable-hierarchical-cache`
- `--hicache-storage-backend mooncake`
- `--hicache-storage-backend-extra-config ...`（可选）

对应配置/校验主要在：

- `python/sglang/srt/server_args.py`
- `python/sglang/srt/environ.py`

其中 `server_args.py` 会做布局兼容修正：当存储后端是 Mooncake（或 Ascend Memcache）且布局为 `layer_first` 时，会根据 I/O 后端自动切到 `page_first` 或 `page_first_direct`。

### 1.2 Scheduler 选择分层缓存实现

`scheduler.py` 在创建 `tree_cache` 时：

- 普通模型走 `HiRadixCache`
- Hybrid SSM 模型走 `HiMambaRadixCache`

这两条路径都会进入 HiCache 控制器（`HiCacheController` / `HybridCacheController`），并在启用 L3 时 attach storage backend。

### 1.3 BackendFactory 实例化 MooncakeStore

路径：

`StorageBackendFactory.create_backend("mooncake", storage_config, mem_pool_host)`

注册位置：

- `python/sglang/srt/mem_cache/storage/backend_factory.py`

Mooncake 对应类：

- `sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.MooncakeStore`

实例化后会调用：

- `storage_backend.register_mem_pool_host(mem_pool_host)`

用于把 Host KV 内存池（L2）注册给 Mooncake 做 zero-copy I/O。

---

## 2) `mooncake_store.py` 的初始化逻辑

核心文件：

- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`

### 2.1 配置加载优先级

`MooncakeBaseStore._load_config` 的优先级：

1. `storage_config.extra_config`（来自 `--hicache-storage-backend-extra-config`）  
2. `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` 指定的 JSON 文件  
3. 环境变量（如 `MOONCAKE_MASTER` / `MOONCAKE_CLIENT` / `MOONCAKE_PROTOCOL` 等）

### 2.2 Store 初始化与连接

`MooncakeStore.__init__` 主要步骤：

1. 导入并创建 `MooncakeDistributedStore`
2. 读取配置并计算 `per_tp_global_segment_size`
3. 可选 `check_server()`（检查 master metrics 接口是否就绪）
4. 按模式 setup：
   - `standalone_storage=True`：`setup_dummy(...)`
   - 否则：`setup(...)`
5. 满足条件时复用共享 Transfer Engine（通过 `parallel_state.get_mooncake_transfer_engine()`）
6. `warmup()`：做一次 put/exist/get 健康自检
7. 初始化 rank 后缀与 key 规则（TP/PP/CP，MLA/MHA，split heads）
8. 初始化存储指标采样缓存（prefetch/backup pgs 与带宽）

### 2.3 Host buffer 注册（zero-copy 基础）

- `register_mem_pool_host(mem_pool_host)`：注册 KV anchor buffer
- `register_mem_host_pool_v2(host_pool, host_pool_name)`：注册 Hybrid 场景的额外 pool（如 MAMBA / INDEXER）

`register_buffer` 最终调用 `store.register_buffer(ptr, size)`，使 Mooncake 能直接按地址读写 Host 内存页面。

---

## 3) 运行期主要数据流向（L3<->L2）

分层缓存语义：

- L1：GPU KV
- L2：Host KV（CPU）
- L3：Mooncake（分布式存储）

`MooncakeStore` 负责 L2 与 L3 的 zero-copy 批量 I/O。

### 3.1 预取路径（L3 -> L2）

1. 请求入队后，`scheduler._prefetch_kvcache()` 触发 `tree_cache.prefetch_from_storage(...)`
2. 控制器 prefetch 线程先做命中探测（`batch_exists` / `batch_exists_v2`）
3. 达到阈值后执行批量读取：
   - KV v1：`batch_get_v1(...)` -> `batch_get_into(...)`
   - Hybrid v2：`batch_get_v2(...)`
4. 读取目标是 Host pool 的页面指针（zero-copy 写入）
5. `check_prefetch_progress()` 将已预取页面并入 host radix 结构，更新 `storage_hit_length`

### 3.2 回写路径（L2 -> L3）

1. 节点满足写策略后，先做 GPU->Host 备份（进入 L2）
2. DMA 确认后，触发 `write_storage(...)` 异步写入 L3
3. `MooncakeStore` 写路径：
   - KV v1：`batch_set_v1(...)`（先 `batch_is_exist`，仅写缺失 key）
   - Hybrid v2：`batch_set_v2(...)`
4. 底层写接口是 `batch_put_from(...)`（按 pointer + size 提交）

### 3.3 key 组织与命中判定要点

- MHA：每页通常拆为 `_k` / `_v` 两个对象
- MLA：每页通常一个 `_k` 对象
- split-head 模式：每页会被拆成更多对象
- v2 额外池（MAMBA/INDEXER）支持命中策略：
  - `ALL_PAGES`
  - `TRAILING_PAGES`
- 最终可用前缀长度为 KV 与各附加池命中边界的最小值

---

以上 1）到 3）覆盖了从启动到运行时读写的主链路，便于继续定位：

- 配置问题（config/env/extra_config）
- attach/setup 问题（初始化/TE 复用/buffer 注册）
- 数据面问题（exists/get/set v1/v2、key 规则、命中边界）
