import argparse
import ast
import sys
import types
from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.ascend_memcache.ascend_memcache_store import (
    AscendMemcacheStore,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore


class _FakeAscendStore:
    def __init__(self):
        self.setup_calls = 0
        self.init_calls = 0
        self.registered = []
        self.closed = False
        self.removed_all = False
        self.batch_get_into_result = [0]
        self.batch_is_exist_result = [1]

    def setup(self, _cfg):
        self.setup_calls += 1
        return 0

    def init(self, _device_id, _init_bm):
        self.init_calls += 1
        return 0

    def put(self, _key, _value):
        return 0

    def is_exist(self, _key):
        return 1

    def get(self, _key):
        return b"\x00" * (4 * 1024)

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def batch_get_into(self, _keys, _ptrs, _sizes):
        return self.batch_get_into_result

    def batch_put_from(self, keys, _ptrs, _sizes):
        return [0] * len(keys)

    def batch_is_exist(self, _keys):
        return self.batch_is_exist_result

    def remove_all(self):
        self.removed_all = True
        return 0

    def close(self):
        self.closed = True


class _FakeMooncakeStore:
    def __init__(self):
        self.setup_calls = 0
        self.registered = []
        self.closed = False
        self.removed_all = False
        self.batch_get_into_result = [8]
        self.batch_is_exist_result = [1]

    def setup(self, *_args):
        self.setup_calls += 1
        return 0

    def setup_dummy(self, *_args):
        self.setup_calls += 1
        return 0

    def put(self, _key, _value):
        return 0

    def is_exist(self, _key):
        return 1

    def get(self, _key):
        return b"\x00" * (4 * 1024)

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def batch_get_into(self, _keys, _ptrs, _sizes):
        return self.batch_get_into_result

    def batch_put_from(self, keys, _ptrs, _sizes):
        return [0] * len(keys)

    def batch_is_exist(self, _keys):
        return self.batch_is_exist_result

    def remove_all(self):
        self.removed_all = True
        return 0

    def close(self):
        self.closed = True


class _FakeLocalConfig:
    pass


def _default_storage_config(extra_config=None):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test-model",
        extra_config=extra_config,
    )


@contextmanager
def _patch_ascend_modules():
    module = types.ModuleType("memcache_hybrid")
    module.DistributedObjectStore = _FakeAscendStore
    module.LocalConfig = _FakeLocalConfig
    old_modules = dict(sys.modules)
    sys.modules["memcache_hybrid"] = module
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(old_modules)


@contextmanager
def _patch_mooncake_modules():
    mooncake_pkg = types.ModuleType("mooncake")
    mooncake_store_mod = types.ModuleType("mooncake.store")
    mooncake_store_mod.MooncakeDistributedStore = _FakeMooncakeStore
    mooncake_pkg.store = mooncake_store_mod
    old_modules = dict(sys.modules)
    sys.modules["mooncake"] = mooncake_pkg
    sys.modules["mooncake.store"] = mooncake_store_mod
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(old_modules)


def _run_common_mapping_checks(store, backend_name: str):
    # extra_backend_tag mapping
    assert store._tag_keys(["a", "b"]) == ["tenant_a_a", "tenant_a_b"]

    # register_buffer mapping
    t = torch.zeros(16, dtype=torch.uint8)
    store.register_buffer(t)
    assert len(store.store.registered) >= 1

    # batch_put_from mapping
    put_ret = store._put_batch_zero_copy_impl(["k1", "k2"], [1, 2], [16, 16])
    assert put_ret == [0, 0]

    # batch_is_exist mapping
    store.store.batch_is_exist_result = [1, 0, -3008]
    ex_ret = store._batch_exist(["k1", "k2", "k3"])
    assert ex_ret == [1, 0, -3008]

    # batch_get_into mapping semantics
    if backend_name == "ascend_memcache":
        store.store.batch_get_into_result = [0, -3, 0]
        get_ret = store._get_batch_zero_copy_impl(
            ["k1", "k2", "k3"], [1, 2, 3], [16, 32, 64]
        )
        assert get_ret == [16, -3, 64]
    else:
        store.store.batch_get_into_result = [16, -3, 64]
        get_ret = store._get_batch_zero_copy_impl(
            ["k1", "k2", "k3"], [1, 2, 3], [16, 32, 64]
        )
        assert get_ret == [16, -3, 64]

    # clear mapping
    store.clear()
    assert store.store.removed_all is True


def _run_ascend_tests():
    with _patch_ascend_modules():
        cfg = _default_storage_config(
            extra_config={"extra_backend_tag": "tenant_a", "check_server": False}
        )
        store = AscendMemcacheStore(cfg)
        _run_common_mapping_checks(store, "ascend_memcache")
        store.close()
        assert store.store is None


def _run_mooncake_tests():
    with _patch_mooncake_modules():
        cfg = _default_storage_config(
            extra_config={
                "master_server_address": "127.0.0.1:50051",
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "tcp",
                "device_name": "",
                "check_server": False,
                "standalone_storage": False,
                "extra_backend_tag": "tenant_a",
            }
        )
        store = MooncakeStore(cfg)
        _run_common_mapping_checks(store, "mooncake")


def _normalize_backend_arg(backend_arg: str):
    if backend_arg is None:
        return ["ascend_memcache", "mooncake"]

    arg = backend_arg.strip()
    lower = arg.lower()
    alias = {
        "ascend": "ascend_memcache",
        "ascend_memcache": "ascend_memcache",
        "ascend_memcahe": "ascend_memcache",  # common typo alias
        "mooncake": "mooncake",
        "all": "all",
        "both": "all",
    }

    if lower in alias:
        val = alias[lower]
        if val == "all":
            return ["ascend_memcache", "mooncake"]
        return [val]

    # Support input like "['ascend', 'mooncake']"
    if arg.startswith("[") and arg.endswith("]"):
        parsed = ast.literal_eval(arg)
        if not isinstance(parsed, list):
            raise ValueError("--backend list format must be a Python list.")
        out = []
        for item in parsed:
            key = str(item).strip().lower()
            if key not in alias or alias[key] == "all":
                raise ValueError(f"Unsupported backend item: {item}")
            mapped = alias[key]
            if mapped not in out:
                out.append(mapped)
        return out

    raise ValueError(
        "Unsupported --backend. Use ascend_memcache | mooncake | all | "
        "ascend | ascend_memcahe | \"['ascend','mooncake']\"."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified lightweight mapping tests for AscendMemcacheStore and MooncakeStore."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="all",
        help="Choose backend(s): ascend_memcache, mooncake, all, ascend, ascend_memcahe, or \"['ascend','mooncake']\"",
    )
    args = parser.parse_args()
    selected = _normalize_backend_arg(args.backend)

    if "ascend_memcache" in selected:
        _run_ascend_tests()
        print("[PASS] ascend_memcache mapping checks")
    if "mooncake" in selected:
        _run_mooncake_tests()
        print("[PASS] mooncake mapping checks")

    print("[PASS] selected backend mapping tests finished")


if __name__ == "__main__":
    main()
