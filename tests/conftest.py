import os

import lightgbm

import fastleaves


def pytest_report_header():
    return f"fastleaves kernel: {fastleaves.isa()}, lightgbm {lightgbm.__version__}"


def pytest_configure():
    # CI sets this under the AVX-512 emulator so a scalar fallback cannot pass unnoticed
    if os.environ.get("FASTLEAVES_REQUIRE_AVX512"):
        assert fastleaves.isa() == "avx512", "FASTLEAVES_REQUIRE_AVX512 is set but AVX-512 is not used"
