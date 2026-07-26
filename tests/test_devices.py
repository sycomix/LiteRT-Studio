from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from litert_studio.core.devices import install_android_apk, parse_adb_devices
from litert_studio.core.errors import ConfigurationError


def test_parse_adb_devices() -> None:
    devices = parse_adb_devices(
        "List of devices attached\n"
        "emulator-5554 device product:sdk model:Pixel_8 device:emu transport_id:1\n"
        "ABC unauthorized transport_id:2\n"
    )

    assert len(devices) == 2
    assert devices[0].model == "Pixel_8"
    assert devices[0].state == "device"
    assert devices[1].state == "unauthorized"


def test_parse_adb_ignores_daemon_messages() -> None:
    assert parse_adb_devices("* daemon started successfully *\n") == ()


def test_install_android_apk_uses_selected_device(tmp_path: Path) -> None:
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    completed = Mock(returncode=0, stdout="Success\n", stderr="")
    with (
        patch("litert_studio.core.devices.shutil.which", return_value="/tools/adb"),
        patch("litert_studio.core.devices.subprocess.run", return_value=completed) as run,
    ):
        result = install_android_apk(apk, "ABC")

    assert result["installed"] is True
    assert run.call_args.args[0] == [
        "/tools/adb",
        "-s",
        "ABC",
        "install",
        "-r",
        str(apk.resolve()),
    ]


def test_install_android_apk_rejects_non_apk(tmp_path: Path) -> None:
    package = tmp_path / "app.zip"
    package.write_bytes(b"zip")
    with pytest.raises(ConfigurationError, match="not an APK"):
        install_android_apk(package)
