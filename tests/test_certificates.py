"""Every certificate in certificates/ passes certificates/verify_certificate.py (needs python-flint)."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("flint")
CERTIFICATES = Path(__file__).resolve().parents[1] / "certificates"
spec = importlib.util.spec_from_file_location("verify_certificate", CERTIFICATES / "verify_certificate.py")
verify_certificate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_certificate)


@pytest.mark.parametrize("path", sorted(CERTIFICATES.glob("*.json")), ids=lambda p: p.stem)
def test_certificate_is_verified(path):
    assert verify_certificate.check(str(path))
