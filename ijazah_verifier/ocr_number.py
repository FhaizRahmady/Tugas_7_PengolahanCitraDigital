"""Cabang kiri pipeline: Area Nomor -> Enhancement -> OCR -> Nomor Ijazah"""
import re
import cv2
import numpy as np
import pytesseract
from .enhancement import METHODS, DEFAULT_METHOD

UPSCALE = 2.0
# Fallback jika label "Nomor ijazah" gagal dilokalisasi (rasio terhadap citra)
FALLBACK_ROI = (0.03, 0.45, 0.89, 0.99)      # x0, x1, y0, y1


def locate_number_roi(gray):
    """Cari kata 'Nomor' pada 30% bawah citra, kembalikan ROI satu baris.
    Return (x0, y0, x1, y1, metode_lokalisasi)."""
    H, W = gray.shape
    y_off = int(0.70 * H)
    sub = gray[y_off:, : int(0.7 * W)]
    try:
        d = pytesseract.image_to_data(sub, config="--psm 11",
                                      output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d["text"]):
            if t.strip().lower().startswith(("nomor", "nomo", "omor")) and d["conf"][i] > 40:
                x, y, w, h = d["left"][i], d["top"][i] + y_off, d["width"][i], d["height"][i]
                return (max(0, x - 15), max(0, int(y - 0.6 * h)),
                        min(W, x + int(0.40 * W)), min(H, int(y + 1.6 * h)), "label")
    except pytesseract.TesseractError:
        pass
    x0, x1, y0, y1 = FALLBACK_ROI
    return int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H), "fallback"


def _digits_from_text(text):
    cands = re.findall(r"\d[\d\-\s]{6,}\d", text)
    cands = [re.sub(r"\s+", "", c) for c in cands]
    return max(cands, key=len) if cands else ""


def ocr_number_roi(roi_gray, method=DEFAULT_METHOD):
    """Enhancement + OCR pada ROI. Return (nomor, ocr_mentah, citra_enhanced)."""
    up = cv2.resize(roi_gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    enh = METHODS[method](up)
    if np.median(enh) < 127:                       # pastikan teks gelap di latar terang
        enh = cv2.bitwise_not(enh)
    enh = cv2.copyMakeBorder(enh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    raw = pytesseract.image_to_string(enh, config="--psm 7")
    num = _digits_from_text(raw)

    # Tahap ke-2: potong bagian di kanan label ("ijazah:") lalu OCR digit-only
    try:
        d = pytesseract.image_to_data(enh, config="--psm 7", output_type=pytesseract.Output.DICT)
        cut = None
        for i, t in enumerate(d["text"]):
            if "ijazah" in t.lower() or t.strip().endswith(":"):
                cut = d["left"][i] + d["width"][i]
        if cut:
            crop = enh[:, max(0, cut - 5):]
            r2 = pytesseract.image_to_string(
                crop, config="--psm 7 -c tessedit_char_whitelist=0123456789-")
            n2 = _digits_from_text(r2)
            if len(n2) >= len(num) or not num:
                num, raw = n2 or num, raw + " | " + r2.strip()
    except pytesseract.TesseractError:
        pass
    return num, raw.strip(), enh


def read_number(gray, method=DEFAULT_METHOD):
    x0, y0, x1, y1, how = locate_number_roi(gray)
    roi = gray[y0:y1, x0:x1]
    num, raw, enh = ocr_number_roi(roi, method)
    return {"number": num, "raw": raw, "roi": (x0, y0, x1, y1),
            "roi_method": how, "roi_img": roi, "enh_img": enh}
