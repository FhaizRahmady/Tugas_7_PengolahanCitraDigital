"""Orkestrasi pipeline lengkap:
Citra -> Grayscale -> Enhancement -> [Area Nomor | Area TTD] -> Hasil Verifikasi"""
import os
import cv2
from .preprocess import auto_orient, to_gray, enhance_global
from .ocr_number import read_number
from .signature import detect_signatures
from .enhancement import DEFAULT_METHOD


def verify(path, out_dir=None, method=DEFAULT_METHOD, save_debug=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")

    img, angle = auto_orient(img)                 # koreksi orientasi
    gray = to_gray(img)                           # 1. Grayscale
    enh = enhance_global(gray)                    # 2. Image Enhancement global

    num = read_number(gray, method)               # cabang kiri (OCR)
    sig = detect_signatures(gray)                 # cabang kanan (TTD)

    result = {
        "file": os.path.basename(path),
        "rotation_applied": angle,
        "nomor_ijazah": num["number"] or "TIDAK TERBACA",
        "tanda_tangan": sig["_final"],
        "detail_ttd": {k: ("PRESENT" if v["present"] else "ABSENT")
                       for k, v in sig.items() if k != "_final"},
        "ocr_mentah": num["raw"],
        "roi_nomor_metode": num["roi_method"],
        "metode_enhancement_ocr": method,
    }
    result["status"] = ("VALID (nomor terbaca & tanda tangan ada)"
                        if num["number"] and sig["_final"] == "PRESENT"
                        else "PERLU DICEK MANUAL")

    if save_debug and out_dir:
        _save_debug(out_dir, os.path.splitext(result["file"])[0],
                    img, gray, enh, num, sig)
    return result


def _save_debug(out_dir, stem, img, gray, enh, num, sig):
    d = os.path.join(out_dir, stem)
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "01_grayscale.png"), gray)
    cv2.imwrite(os.path.join(d, "02_enhanced.png"), enh)
    cv2.imwrite(os.path.join(d, "03_roi_nomor.png"), num["roi_img"])
    cv2.imwrite(os.path.join(d, "04_nomor_enhanced_untuk_ocr.png"), num["enh_img"])
    vis = img.copy()
    x0, y0, x1, y1 = num["roi"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 4)
    cv2.putText(vis, f"OCR: {num['number']}", (x0, max(30, y0 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    for name, v in sig.items():
        if name == "_final":
            continue
        xa, ya, xb, yb = v["roi"]
        col = (0, 160, 0) if v["present"] else (0, 0, 255)
        cv2.rectangle(vis, (xa, ya), (xb, yb), col, 4)
        cv2.putText(vis, f"{name}: {'PRESENT' if v['present'] else 'ABSENT'}",
                    (xa, ya - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        cv2.imwrite(os.path.join(d, f"05_ttd_{name}_threshold.png"), v["bw"])
        cv2.imwrite(os.path.join(d, f"06_ttd_{name}_morphology.png"), v["closed"])
        if v["best"]:
            bx, by, bw_, bh = v["best"]["bbox"]
            cv2.rectangle(vis, (xa + bx, ya + by), (xa + bx + bw_, ya + by + bh), col, 2)
    cv2.imwrite(os.path.join(d, "00_hasil_verifikasi.jpg"), vis)
