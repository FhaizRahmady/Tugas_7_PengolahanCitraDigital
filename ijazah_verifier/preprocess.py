"""Tahap 1-3 pipeline: Citra -> Grayscale -> Image Enhancement (global)
plus koreksi orientasi otomatis (scan ijazah sering terputar 90 derajat)."""
import cv2
import pytesseract


def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def normalize_background(gray, ksize=51):
    """Hilangkan pola guilloche / warna kertas kekuningan dengan membagi
    citra dengan estimasi latar (median blur besar). Hasil: latar ~putih."""
    bg = cv2.medianBlur(gray, ksize)
    return cv2.divide(gray, bg, scale=255)


def enhance_global(gray):
    """Enhancement global: normalisasi latar + CLAHE (kontras lokal)."""
    norm = normalize_background(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def _orientation_score(gray):
    """Skor OCR cepat: jumlah kata kunci ijazah yang terbaca."""
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        txt = pytesseract.image_to_string(small, config="--psm 11").lower()
    except pytesseract.TesseractError:
        return 0
    keys = ["universitas", "indonesia", "ijazah", "memberikan", "kepada",
            "nomor", "rektor", "dekan", "fakultas", "program", "magister"]
    return sum(k in txt for k in keys)


def auto_orient(img_bgr):
    """Coba rotasi 0/90/180/270 (hanya yang landscape), pilih yang paling
    banyak menghasilkan kata kunci ijazah.
    Return (citra_terkoreksi, sudut_putar_searah_jarum_jam)."""
    rots = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    best, best_s, best_a = img_bgr, -1, 0
    for ang, code in rots.items():
        cand = img_bgr if code is None else cv2.rotate(img_bgr, code)
        if cand.shape[1] < cand.shape[0]:      # ijazah = landscape
            continue
        s = _orientation_score(to_gray(cand))
        if s > best_s:
            best, best_s, best_a = cand, s, ang
    return best, best_a
