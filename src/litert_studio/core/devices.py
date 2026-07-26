from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AndroidDevice:
    serial: str
    state: str
    product: str | None
    model: str | None
    device: str | None
    transport_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AndroidDiscovery:
    adb_available: bool
    devices: tuple[AndroidDevice, ...]
    issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_android_devices() -> AndroidDiscovery:
    adb = shutil.which("adb")
    if adb is None:
        return AndroidDiscovery(False, (), "Android platform-tools (adb) not found")
    try:
        result = subprocess.run(
            [adb, "devices", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AndroidDiscovery(True, (), str(exc))
    if result.returncode != 0:
        return AndroidDiscovery(True, (), result.stderr.strip() or "adb failed")
    return AndroidDiscovery(True, parse_adb_devices(result.stdout))


def parse_adb_devices(output: str) -> tuple[AndroidDevice, ...]:
    devices: list[AndroidDevice] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith(("List of devices", "*")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        attributes = {
            key: value for part in parts[2:] if ":" in part for key, value in [part.split(":", 1)]
        }
        devices.append(
            AndroidDevice(
                serial=parts[0],
                state=parts[1],
                product=attributes.get("product"),
                model=attributes.get("model"),
                device=attributes.get("device"),
                transport_id=attributes.get("transport_id"),
            )
        )
    return tuple(devices)
