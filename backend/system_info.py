"""Hardware detection and model-tier recommendation."""
from __future__ import annotations

import psutil

try:
    import pynvml as _pynvml
    _NVML_OK = True
except ImportError:
    _NVML_OK = False

try:
    import cpuinfo as _cpuinfo
    _CPUINFO_OK = True
except ImportError:
    _CPUINFO_OK = False


def _gpu_info() -> dict:
    if not _NVML_OK:
        return {"gpu_name": None, "gpu_vram_total_gb": None, "gpu_vram_free_gb": None}
    try:
        _pynvml.nvmlInit()
        handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
        name = _pynvml.nvmlDeviceGetName(handle)
        mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_name": name if isinstance(name, str) else name.decode(),
            "gpu_vram_total_gb": round(mem.total / 1e9, 1),
            "gpu_vram_free_gb": round(mem.free / 1e9, 1),
        }
    except Exception:
        return {"gpu_name": None, "gpu_vram_total_gb": None, "gpu_vram_free_gb": None}


def _cpu_has_avx2() -> bool:
    if not _CPUINFO_OK:
        return False
    try:
        flags = _cpuinfo.get_cpu_info().get("flags", [])
        return "avx2" in flags
    except Exception:
        return False


def _recommend(ram_available_gb: float, gpu_vram_free_gb: float | None) -> tuple[str, str]:
    if gpu_vram_free_gb is not None:
        if gpu_vram_free_gb >= 10:
            return "12b", f"GPU VRAM: {gpu_vram_free_gb} GB free"
        if gpu_vram_free_gb >= 6:
            return "9b", f"GPU VRAM: {gpu_vram_free_gb} GB free"
    if ram_available_gb >= 12:
        return "12b", f"RAM: {ram_available_gb} GB available"
    if ram_available_gb >= 6:
        return "9b", f"RAM: {ram_available_gb} GB available"
    return "4b", f"RAM: {ram_available_gb} GB available (low)"


def get_system_info() -> dict:
    vm = psutil.virtual_memory()
    ram_total = round(vm.total / 1e9, 1)
    ram_available = round(vm.available / 1e9, 1)
    cpu_physical = psutil.cpu_count(logical=False) or 1
    cpu_logical = psutil.cpu_count(logical=True) or 1
    gpu = _gpu_info()
    recommended, reason = _recommend(ram_available, gpu["gpu_vram_free_gb"])
    return {
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_available,
        "cpu_cores_physical": cpu_physical,
        "cpu_cores_logical": cpu_logical,
        "cpu_has_avx2": _cpu_has_avx2(),
        **gpu,
        "recommended_model": recommended,
        "recommendation_reason": reason,
    }
