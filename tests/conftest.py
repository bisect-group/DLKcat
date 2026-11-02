def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; run with verify_equivalence.sh --gpu")
