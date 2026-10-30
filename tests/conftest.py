r"""Shared fixtures. pytest's default tmp_path basetemp can be ACL-locked on this
machine (PermissionError on %TEMP%\pytest-of-*), so we root test temp dirs in the
pre-approved opencode temp directory instead."""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

_OPENCODE_TMP = os.environ.get("OPENCODE_TMP",
                               r"C:\Users\PARKBU~1\AppData\Local\Temp\opencode")


@pytest.fixture
def tdir():
    root = os.path.join(_OPENCODE_TMP, "nx_tests")
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(dir=root)
    yield d
    shutil.rmtree(d, ignore_errors=True)