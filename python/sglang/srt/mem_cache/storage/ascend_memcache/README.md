# Ascend MemCache HiCache 后端

## 一、改动总结

下面是本次接入 `ascend_memcache` 时已完成的改动与文件清单。

### 1) 核心目标

- 参考 Mooncake 的实现思路接入 Ascend MemCache，但采用**平行实现**：
  - `AscendMemcacheStore` 直接继承 `HiCacheStorage`
  - 不继承 `MooncakeStore` / `MooncakeBaseStore`
- 新增内置存储后端名：`ascend_memcache`
- 保持 HiCache 现有 zero-copy / v1 / v2 流程兼容

### 2) 主要代码修改

- 新增 Ascend MemCache 存储后端实现：
  - 使用 `memcache_hybrid.DistributedObjectStore` 完成 `setup/init`
  - 保留与 Mooncake 一致的 key 组织策略（MHA / MLA / split-heads）
  - 支持 v1/v2 批量接口（`batch_get_v1` / `batch_set_v1`、`batch_get_v2` / `batch_set_v2`）
  - 适配返回码差异：`memcache_hybrid.batch_get_into` 成功返回 `0`，内部转换为 HiCache 读成功语义（正数）
  - 支持 `extra_backend_tag`、`check_server`、`metrics_url`

- 接入后端工厂与参数系统：
  - 在后端工厂注册 `ascend_memcache`
  - 让 cache controller 将该后端纳入 zero-copy 路径
  - `server_args` 增加 `--hicache-storage-backend ascend_memcache` 选项
  - 将 Mooncake 的 `layer_first` 兼容处理扩展到 `ascend_memcache`

- 新增环境变量：
  - `SGLANG_HICACHE_MEMCACHE_CONFIG_PATH`

### 3) 文档改动

- 本文件补充了：
  - 快速使用示例
  - 配置来源优先级
  - SGLang wrapper 控制参数说明
  - Mooncake -> Memcache 参数映射
  - 基础排障说明

### 4) UT 放置策略

- UT 统一放在 `ascend_memcache` 目录下。

### 5) 文件清单

#### 新增文件

- `python/sglang/srt/mem_cache/storage/ascend_memcache/ascend_memcache_store.py`
- `python/sglang/srt/mem_cache/storage/ascend_memcache/__init__.py`
- `python/sglang/srt/mem_cache/storage/ascend_memcache/README.md`
- `python/sglang/srt/mem_cache/storage/ascend_memcache/test_ascend_memcache_store.py`

#### 修改文件

- `python/sglang/srt/environ.py`
- `python/sglang/srt/mem_cache/storage/backend_factory.py`
- `python/sglang/srt/managers/cache_controller.py`
- `python/sglang/srt/server_args.py`

## 二、后端说明

本目录为 SGLang 提供了一个内置 HiCache L3 存储后端：

- 后端名称：`ascend_memcache`
- 实现类：`AscendMemcacheStore`
- 依赖包：`memcache_hybrid`

实现方式与 Mooncake 后端是平行关系（遵循同一套 `HiCacheStorage` 契约与 key 组织策略）

## 三、快速开始

先安装 Python 绑定：

```bash
pip install memcache_hybrid
```

先启动 MemCache 的 MetaService / LocalService 集群，再启动 SGLang：

```bash
python -m sglang.launch_server \
  --model-path /path/to/model \
  --enable-hierarchical-cache \
  --hicache-storage-backend ascend_memcache \
  --hicache-storage-backend-extra-config '{
    "meta_service_url": "tcp://127.0.0.1:5000",
    "config_store_url": "tcp://127.0.0.1:6000",
    "protocol": "host_tcp",
    "dram_size": "8GB",
    "hbm_size": "0",
    "device_id": 0,
    "init_bm": true
  }'
```

## 四、配置来源

`AscendMemcacheStore` 会合并以下两处配置：

1. `SGLANG_HICACHE_MEMCACHE_CONFIG_PATH`（JSON 文件）
2. `--hicache-storage-backend-extra-config`（JSON 字符串或文件）

如果两者同时设置，`extra-config` 的优先级更高，会覆盖文件中的同名字段。

## 五、控制字段（SGLang wrapper）

下列字段由 SGLang wrapper 消费，不会透传给 `memcache_hybrid.LocalConfig`：

- `device_id`（默认：`0`）
- `init_bm`（默认：`true`）
- `check_server`（默认：`false`）
- `metrics_url` / `memcache_metrics_url`（用于启动就绪检查）
- `extra_backend_tag`（为全部 cache key 增加统一前缀）

## 六、Mooncake 到 Memcache 参数映射

如果你从 `--hicache-storage-backend mooncake` 迁移过来，可参考：

- **后端切换**
  - Mooncake：`--hicache-storage-backend mooncake`
  - Memcache：`--hicache-storage-backend ascend_memcache`
- **服务端地址**
  - Mooncake 常见字段：`master_server_address` + `metadata_server`
  - Memcache 常见字段：`meta_service_url` + `config_store_url`（`LocalConfig`）
- **网络协议**
  - Mooncake 示例常见：`protocol=rdma`
  - Memcache 使用自己的协议值（例如 `host_tcp`、`host_rdma`、`device_sdma`）
- **设备选择**
  - Mooncake 常见：`device_name`
  - Memcache wrapper 使用：`device_id`
- **服务就绪检查**
  - Mooncake 使用内置 master metrics 轮询
  - Memcache 使用 wrapper 字段：`check_server=true` + `metrics_url`（或 `memcache_metrics_url`）
- **命名空间隔离**
  - 两者都支持在 SGLang 的 extra config 中使用 `extra_backend_tag`

## 七、排障建议

- **`memcache_hybrid` 导入失败**
  - 在与 SGLang 相同的 Python 环境安装：`pip install memcache_hybrid`
- **`setup` / `init` 失败**
  - 检查 `meta_service_url` 和 `config_store_url` 是否从 SGLang 所在机器可达
  - 确认 MemCache 服务已在 SGLang 启动前拉起
- **读路径持续 miss**
  - 检查生产者与消费者是否使用相同的 `extra_backend_tag`
  - 确认两侧 TP / PP / CP 配置兼容（key 后缀包含 rank 信息）

## 八、补充说明

- MemCache 的 `batch_get_into` 在成功时返回 `0`。后端内部会将其适配成 HiCache 的读成功语义（正数），以兼容既有后处理逻辑。
- 布局兼容遵循 SGLang 中 Mooncake 的处理方式：当参数检查触发时，`layer_first` 会被改写为 `page_first*` 兼容布局。
