"""Jalankan: python -m pytest -q   (atau: python tests/test_pipeline.py)"""
import os
import sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ijazah_verifier.pipeline import verify
from ijazah_verifier.preprocess import auto_orient, to_gray
from ijazah_verifier.signature import SIGN_REGIONS
from evaluate_cer import cer

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "ijazah_001.jpg")
GT = "571012022000056"


def test_cer_function():
    assert cer("123", "123") == 0
    assert abs(cer("124", "123") - 1 / 3) < 1e-9


def test_positive_sample():
    r = verify(SAMPLE, save_debug=False)
    assert r["nomor_ijazah"] == GT
    assert r["tanda_tangan"] == "PRESENT"


def test_negative_signature_removed(tmp_path):
    """Area tanda tangan diganti kertas kosong -> harus ABSENT."""
    img, _ = auto_orient(cv2.imread(SAMPLE))
    H, W = img.shape[:2]
    blank = img[int(.30 * H):int(.34 * H), int(.02 * W):int(.12 * W)]
    for (x0, x1, y0, y1) in SIGN_REGIONS.values():
        xa, xb, ya, yb = int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)
        img[ya:yb, xa:xb] = cv2.resize(blank, (xb - xa, yb - ya))
    p = str(tmp_path / "neg.jpg")
    cv2.imwrite(p, img)
    r = verify(p, save_debug=False)
    assert r["tanda_tangan"] == "ABSENT"
    assert r["nomor_ijazah"] == GT


def test_rotated_input():
    """Input terputar 90/180/270 tetap terbaca."""
    img = cv2.imread(SAMPLE)
    for code in (cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        p = "/tmp/_rot_test.jpg"
        cv2.imwrite(p, cv2.rotate(img, code))
        assert verify(p, save_debug=False)["nomor_ijazah"] == GT


if __name__ == "__main__":
    import tempfile, pathlib
    test_cer_function(); test_positive_sample()
    test_negative_signature_removed(pathlib.Path(tempfile.mkdtemp()))
    test_rotated_input()
    print("Semua test lulus")
