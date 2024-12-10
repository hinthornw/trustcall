import pytest


@pytest.fixture(scope="session", autouse=True)
def anyio_backend():
    return "asyncio"
