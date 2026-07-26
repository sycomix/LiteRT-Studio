from litert_studio.core.devices import parse_adb_devices


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
