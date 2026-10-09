"""Cabang kanan pipeline: Area TTD -> Thresholding -> Morphology -> Signature Detection.

Ide utama: tanda tangan = goresan tinta tangan yang (a) menyambung membentuk
komponen besar setelah closing morfologi, sedangkan (b) teks cetak berupa
komponen kecil/seragam yang dibuang. Keputusan memakai ukuran bbox, luas tinta
dan jumlah komponen goresan besar."""
import cv2
import numpy as np
from .preprocess import normalize_background

# Layout ijazah UI (rasio x0, x1, y0, y1 terhadap citra landscape tegak).
# ROI sengaja dipilih di antara teks cetak "Rektor/Dekan" dan nama pejabat.
SIGN_REGIONS = {
    "Rektor": (0.10, 0.42, 0.715, 0.835),
    "Dekan":  (0.60, 0.90, 0.670, 0.835),
}

INK_MAX_GRAY = 150      # piksel ternormalisasi harus < 150 agar dihitung tinta
MIN_W_RATIO = 0.07     # lebar bbox komponen min. 7% lebar citra
MIN_INK_PIXELS_RATIO = 0.004  # luas piksel tinta komponen / luas ROI


def threshold_roi(roi_gray):
    """Thresholding: normalisasi latar -> blur -> Otsu invers (tinta = putih)."""
    n = normalize_background(roi_gray, 51)
    n = cv2.GaussianBlur(n, (5, 5), 0)
    otsu, _ = cv2.threshold(n, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Batas kontras absolut: pada ROI kosong Otsu akan membelah pola latar
    # (false positive). Tinta asli jauh lebih gelap dari latar ternormalisasi.
    t = min(otsu, INK_MAX_GRAY)
    return (n < t).astype(np.uint8) * 255


def morphology(bw):
    """Opening kecil (buang noise guilloche) lalu closing besar (sambung goresan)."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    o = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(o, cv2.MORPH_CLOSE, k_close), o


def detect_in_roi(roi_gray, img_w):
    bw = threshold_roi(roi_gray)
    closed, opened = morphology(bw)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    roi_area = roi_gray.shape[0] * roi_gray.shape[1]
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        ink = int(np.count_nonzero(opened[lab == i]))   # tinta asli, bukan hasil closing
        wide = w >= MIN_W_RATIO * img_w
        enough_ink = ink >= MIN_INK_PIXELS_RATIO * roi_area
        stroke_like = h >= 0.12 * roi_gray.shape[0] or w >= 0.20 * roi_gray.shape[1]
        if wide and enough_ink and stroke_like:
            if best is None or ink > best["ink"]:
                best = {"bbox": (int(x), int(y), int(w), int(h)), "ink": ink}
    return best, bw, closed


def detect_signatures(gray):
    """Return dict per region + keputusan akhir."""
    H, W = gray.shape
    res = {}
    for name, (x0, x1, y0, y1) in SIGN_REGIONS.items():
        xa, xb, ya, yb = int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)
        roi = gray[ya:yb, xa:xb]
        best, bw, closed = detect_in_roi(roi, W)
        res[name] = {"present": best is not None, "best": best,
                     "roi": (xa, ya, xb, yb), "roi_img": roi,
                     "bw": bw, "closed": closed}
    # Tanda tangan pimpinan dianggap ADA bila minimal satu (Rektor/Dekan) terdeteksi
    res["_final"] = "PRESENT" if any(v["present"] for k, v in res.items()) else "ABSENT"
    return res
