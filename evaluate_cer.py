"""Evaluasi metode enhancement berdasarkan CER (Character Error Rate).

CER = (S + D + I) / N   (jarak Levenshtein / panjang ground truth)

Karena citra scan asli biasanya sudah bersih, setiap ROI nomor diuji pada
beberapa kondisi degradasi sintetis (noise, blur, kontras rendah, bayangan,
JPEG, dll) supaya perbedaan antar metode enhancement terlihat.
Penggunaan:  python evaluate_cer.py [--gt samples/ground_truth.csv]
"""
import argparse
import csv
import os
import cv2
import numpy as np
from ijazah_verifier.preprocess import auto_orient, to_gray
from ijazah_verifier.ocr_number import locate_number_roi, ocr_number_roi
from ijazah_verifier.enhancement import METHODS

SEED = 42


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, gt):
    return levenshtein(pred, gt) / max(1, len(gt))


# ---------------- degradasi sintetis ----------------
def d_clean(g, rng):
    return g

def d_noise(g, rng):
    return np.clip(g + rng.normal(0, 28, g.shape), 0, 255).astype(np.uint8)

def d_blur(g, rng):
    return cv2.GaussianBlur(g, (0, 0), 2.2)

def d_lowcontrast(g, rng):
    return np.clip(0.35 * g.astype(np.float32) + 110, 0, 255).astype(np.uint8)

def d_shadow(g, rng):
    h, w = g.shape
    grad = np.linspace(1.0, 0.35, w, dtype=np.float32)[None, :]
    return np.clip(g.astype(np.float32) * grad, 0, 255).astype(np.uint8)

def d_saltpepper(g, rng):
    o = g.copy()
    m = rng.random(g.shape)
    o[m < 0.02] = 0
    o[m > 0.98] = 255
    return o

def d_jpeg(g, rng):
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 12])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

def d_combo(g, rng):
    x = d_shadow(g, rng)
    x = d_blur(x, rng)
    return d_noise(x, rng)

DEGRADATIONS = {
    "clean": d_clean, "gaussian_noise": d_noise, "blur": d_blur,
    "low_contrast": d_lowcontrast, "uneven_shadow": d_shadow,
    "salt_pepper": d_saltpepper, "jpeg_q12": d_jpeg, "combo(shadow+blur+noise)": d_combo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="samples/ground_truth.csv")
    ap.add_argument("--out", default="outputs/cer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = os.path.dirname(a.gt)

    rows = []   # (image, degradasi, metode, pred, gt, cer)
    for rec in csv.DictReader(open(a.gt)):
        img = cv2.imread(os.path.join(base, rec["image"]))
        img, _ = auto_orient(img)
        gray = to_gray(img)
        x0, y0, x1, y1, _ = locate_number_roi(gray)
        roi = gray[y0:y1, x0:x1]
        gt = rec["nomor_ijazah"]
        for dname, dfn in DEGRADATIONS.items():
            rng = np.random.default_rng(SEED)
            deg = dfn(roi, rng)
            if dname != "clean":
                cv2.imwrite(os.path.join(a.out, f"{os.path.splitext(rec['image'])[0]}_{dname.split('(')[0]}.png"), deg)
            for mname in METHODS:
                try:
                    pred, _, _ = ocr_number_roi(deg, mname)
                except Exception:
                    pred = ""
                rows.append((rec["image"], dname, mname, pred, gt, cer(pred, gt)))

    with open(os.path.join(a.out, "cer_detail.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "degradation", "method", "prediction", "ground_truth", "CER"])
        w.writerows(rows)

    methods, degs = list(METHODS), list(DEGRADATIONS)
    mean = {m: np.mean([r[5] for r in rows if r[2] == m]) for m in methods}
    table = {(m, d): np.mean([r[5] for r in rows if r[2] == m and r[1] == d])
             for m in methods for d in degs}

    lines = ["| Metode | " + " | ".join(degs) + " | **Rata-rata CER** |",
             "|---|" + "---|" * (len(degs) + 1)]
    for m in sorted(methods, key=lambda k: mean[k]):
        lines.append(f"| {m} | " + " | ".join(f"{table[(m, d)]:.3f}" for d in degs)
                     + f" | **{mean[m]:.3f}** |")
    md = "\n".join(lines)
    best = min(mean, key=mean.get)
    md += f"\n\n**Metode terbaik (rata-rata CER terendah): `{best}` ({mean[best]:.3f})**\n"
    open(os.path.join(a.out, "cer_table.md"), "w").write(md)
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = sorted(methods, key=lambda k: mean[k])
        plt.figure(figsize=(9, 4.5))
        bars = plt.barh(order[::-1], [mean[m] for m in order[::-1]], color="#4C78A8")
        plt.xlabel("Rata-rata CER (lebih rendah = lebih baik)")
        plt.title("Perbandingan metode enhancement berdasarkan CER")
        for b, m in zip(bars, order[::-1]):
            plt.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{mean[m]:.3f}", va="center")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "cer_chart.png"), dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
