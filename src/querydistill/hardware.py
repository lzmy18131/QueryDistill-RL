"""Hardware / environment doctor.

Reports machine facts and performs the project's D-drive cache audit. The
doctor never moves files and never changes the system environment.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings, audit_cache_paths
from .utils import package_version

_WSL_OSRELEASE = Path("/proc/sys/kernel/osrelease")
_WSL_PROCVERSION = Path("/proc/version")


def detect_wsl() -> tuple[bool, str]:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True, f"WSL distro: {os.environ.get('WSL_DISTRO_NAME')}"
    if _WSL_OSRELEASE.exists():
        content = _WSL_OSRELEASE.read_text(encoding="utf-8", errors="ignore").lower()
        if "microsoft" in content:
            return True, f"kernel: {content.strip()}"
    if _WSL_PROCVERSION.exists():
        content = _WSL_PROCVERSION.read_text(encoding="utf-8", errors="ignore").lower()
        if "microsoft" in content:
            return True, "proc version contains microsoft"
    return False, "native OS (not detected as WSL)"


def _nvidia_smi() -> dict[str, str] | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        proc = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        return None
    return {
        "name": parts[0],
        "memory_total_mib": parts[1],
        "memory_free_mib": parts[2],
        "driver_version": parts[3],
    }


def _torch_gpu() -> dict | None:
    try:
        import torch
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.get_device_properties(0)
        total_mib = device.total_memory / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        free_mib = total_mib - reserved
        major, minor = device.major, device.minor
        return {
            "name": device.name,
            "memory_total_mib": f"{total_mib:.0f}",
            "memory_free_mib": f"{free_mib:.0f}",
            "capability": f"{major}.{minor}",
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception:  # noqa: BLE001 - best effort
        return None


@dataclass
class HardwareReport:
    os: str
    os_release: str
    wsl_detected: bool
    wsl_detail: str
    python_version: str
    python_executable: str
    cpu_model: str
    cpu_cores_logical: int
    cpu_cores_physical: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_name: str
    gpu_vram_total_mib: float
    gpu_vram_free_mib: float
    nvidia_driver: str
    cuda_available: bool
    cuda_device_count: int
    torch_version: str
    torch_cuda_version: str
    bf16_supported: bool
    transformers_version: str
    trl_version: str
    peft_version: str
    bitsandbytes_version: str
    sqlglot_version: str
    gptqmodel_version: str
    vllm_version: str
    project_root: str
    project_free_gb: float
    cache_root: str
    cache_free_gb: float
    cache_audit: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def probe(settings: Settings | None = None) -> HardwareReport:
    settings = settings or Settings.load()
    settings.ensure_directories()
    is_wsl, wsl_detail = detect_wsl()
    smi = _nvidia_smi()
    torch_gpu = _torch_gpu()

    try:
        import psutil

        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_free = psutil.virtual_memory().available / (1024**3)
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        ram_total = ram_free = 0.0
        physical = logical = os.cpu_count() or 0

    def version(name: str) -> str:
        value = package_version(name)
        return value if value is not None else "not installed"

    def free_gb(path: Path) -> float:
        try:
            return shutil.disk_usage(path).free / (1024**3)
        except OSError:
            return -1.0

    torch_version = version("torch")
    try:
        import torch

        torch_cuda = torch.version.cuda or "not available"
        cuda_available = bool(torch.cuda.is_available())
        device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception:  # noqa: BLE001 - torch optional here
        torch_cuda = "not available"
        cuda_available = False
        device_count = 0

    gpu_source = torch_gpu or smi or {}
    bf16 = bool(gpu_source.get("bf16_supported")) or bool(
        smi and "4060" in gpu_source.get("name", "")
    )
    warnings: list[str] = []
    if torch_version == "not installed":
        warnings.append("torch is not installed in the active environment")
    if not cuda_available and not smi:
        warnings.append("no CUDA device / nvidia-smi could not be queried")
    if "cpu" in torch_version.lower() and smi:
        warnings.append(
            "active torch build is CPU-only; CUDA GPU is visible to nvidia-smi but not to torch"
        )

    audit = audit_cache_paths(settings)
    warnings.extend(
        f"[{item.level.upper()}] {item.name}: {item.detail}" for item in audit if item.level != "ok"
    )

    return HardwareReport(
        os=f"{platform.system()} {platform.machine()}",
        os_release=platform.release(),
        wsl_detected=is_wsl,
        wsl_detail=wsl_detail,
        python_version=platform.python_version(),
        python_executable=os.sys.executable,
        cpu_model=platform.processor() or "unknown",
        cpu_cores_logical=int(logical or 0),
        cpu_cores_physical=int(physical or 0),
        ram_total_gb=round(ram_total, 2),
        ram_free_gb=round(ram_free, 2),
        gpu_name=gpu_source.get("name", "not detected"),
        gpu_vram_total_mib=float(gpu_source.get("memory_total_mib", 0) or 0),
        gpu_vram_free_mib=float(gpu_source.get("memory_free_mib", 0) or 0),
        nvidia_driver=gpu_source.get("driver_version", "not detected"),
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda,
        bf16_supported=bf16,
        transformers_version=version("transformers"),
        trl_version=version("trl"),
        peft_version=version("peft"),
        bitsandbytes_version=version("bitsandbytes"),
        sqlglot_version=version("sqlglot"),
        gptqmodel_version=version("gptqmodel"),
        vllm_version=version("vllm"),
        project_root=str(settings.project_root),
        project_free_gb=round(free_gb(settings.project_root), 2),
        cache_root=str(settings.cache_root),
        cache_free_gb=round(free_gb(settings.cache_root), 2),
        cache_audit=[asdict(item) for item in audit],
        warnings=warnings,
    )


def format_report(report: HardwareReport) -> str:
    lines = [
        "QueryDistill-RL hardware doctor",
        "===============================",
        f"OS                     : {report.os} ({report.os_release})",
        f"WSL detection          : {report.wsl_detected} - {report.wsl_detail}",
        f"Python                 : {report.python_version}",
        f"Python executable      : {report.python_executable}",
        f"CPU                    : {report.cpu_model} "
        f"({report.cpu_cores_physical} physical / {report.cpu_cores_logical} logical cores)",
        f"RAM total/free         : {report.ram_total_gb:.2f} GiB / {report.ram_free_gb:.2f} GiB",
        f"GPU                    : {report.gpu_name}",
        f"GPU VRAM total/free    : {report.gpu_vram_total_mib:.0f} MiB / "
        f"{report.gpu_vram_free_mib:.0f} MiB",
        f"NVIDIA driver          : {report.nvidia_driver}",
        f"CUDA availability      : {report.cuda_available} ({report.cuda_device_count} device(s))",
        f"torch                  : {report.torch_version} (CUDA: {report.torch_cuda_version})",
        f"bf16 support           : {report.bf16_supported}",
        f"transformers           : {report.transformers_version}",
        f"trl                    : {report.trl_version}",
        f"peft                   : {report.peft_version}",
        f"bitsandbytes           : {report.bitsandbytes_version}",
        f"sqlglot                : {report.sqlglot_version}",
        f"gptqmodel              : {report.gptqmodel_version}",
        f"vllm                   : {report.vllm_version}",
        f"project root           : {report.project_root}",
        f"project root free      : {report.project_free_gb:.2f} GiB",
        f"cache root             : {report.cache_root}",
        f"cache root free        : {report.cache_free_gb:.2f} GiB",
    ]
    for item in report.cache_audit:
        lines.append(f"path audit [{item['level'].upper():7}] : {item['name']} - {item['detail']}")
    if report.warnings:
        lines.append("WARNINGS:")
        lines.extend(f"  - {warning}" for warning in report.warnings)
    else:
        lines.append("WARNINGS: none")
    return "\n".join(lines)
