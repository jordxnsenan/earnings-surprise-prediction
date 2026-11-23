
from importlib import import_module

def test_package_imports():
    package = import_module("earnings_surprise")

    assert package.__name__ == "earnings_surprise"