"""Default every automated test to a simulated driver, with no real telemetry/unlock."""
import pytest


@pytest.fixture(autouse=True)
def no_physical_robot(monkeypatch):
    monkeypatch.setenv('TROVA_ARM', 'sim')
    monkeypatch.setenv('TROVA_ARM_READ', 'off')
    monkeypatch.setenv('TROVA_HARDWARE_UNLOCK', '')
