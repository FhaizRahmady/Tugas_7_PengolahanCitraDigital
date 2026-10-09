"""Kumpulan metode enhancement untuk area nomor ijazah.
Semua fungsi: citra gray uint8 -> citra uint8 (gray atau biner)
Dibandingkan lewat nilai CER di evaluate_cer.py."""
import cv2
import numpy as np
from .preprocess import normalize_background


def m_none(g):
    return g


def m_hist_eq(g):
    return cv2.equalizeHist(g)


def m_clahe(g):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def m_gauss_otsu(g):
    b = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def m_adaptive(g):
    b = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def m_unsharp(g):
    b = cv2.GaussianBlur(g, (0, 0), 3)
    return cv2.addWeighted(g, 1.8, b, -0.8, 0)


def m_bgnorm_otsu(g):
    n = normalize_background(g, 51)
    n = cv2.GaussianBlur(n, (3, 3), 0)
    return cv2.threshold(n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _despeckle(bw_text_black, min_area=30):
    """Hapus komponen terhubung kecil (bintik pola guilloche / noise).
    Input: teks hitam (0) di latar putih (255)."""
    inv = (bw_text_black == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    out = np.where(keep[lab], 0, 255).astype(np.uint8)
    return out


def m_full(g):
    """Kombinasi (usulan): denoise median -> normalisasi latar -> blur halus
    -> threshold Otsu -> despeckle komponen kecil
    -> closing kecil agar goresan putus tersambung."""
    d = cv2.medianBlur(g, 3)
    n = normalize_background(d, 51)
    n = cv2.GaussianBlur(n, (0, 0), 1.5)
    otsu, _ = cv2.threshold(n, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = np.where(n < otsu, 0, 255).astype(np.uint8)
    bw = _despeckle(bw, 30)
    inv = cv2.morphologyEx(255 - bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return 255 - inv


METHODS = {
    "1_grayscale_saja": m_none,
    "2_histogram_equalization": m_hist_eq,
    "3_clahe": m_clahe,
    "4_gaussian_otsu": m_gauss_otsu,
    "5_adaptive_threshold": m_adaptive,
    "6_unsharp_mask": m_unsharp,
    "7_bgnorm_otsu": m_bgnorm_otsu,
    "8_kombinasi_full": m_full,
}
DEFAULT_METHOD = "8_kombinasi_full"
