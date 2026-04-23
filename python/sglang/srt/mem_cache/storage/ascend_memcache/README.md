# Ascend MemCache 文件索引

此 README 仅保留以下两个文件的说明：

- `python/sglang/srt/mem_cache/storage/ascend_memcache/ascend_memcache_store.py`
- `python/sglang/srt/mem_cache/storage/test_storage_backend_mapping_unit.py`

## ascend_memcache_store.py

`AscendMemcacheStore` 是 `ascend_memcache` 后端的核心实现，负责：

- 对接 `memcache_hybrid.DistributedObjectStore` 完成 `setup/init/close`
- 实现 HiCache 的 zero-copy 读写接口（含 v1/v2 路径）
- 处理 MemCache 返回码与 HiCache 语义的适配（例如 `batch_get_into` 成功码转换）
- 仿照Mooncake的实现方式

## test_storage_backend_mapping_unit.py

该测试文件用于验证存储后端映射和注册行为，重点覆盖：

- `ascend_memcache` 在后端工厂中的名称映射是否正确
- 配置参数解析后是否落到预期后端
- 与现有后端（如 `mooncake`）并存时的选择逻辑是否稳定
- 关键错误路径的单元行为是否符合预期
