# Verifikasi Ijazah — OCR Nomor Ijazah & Deteksi Tanda Tangan

Mini project **Pengolahan Citra Digital**

| | |
|---|---|
| **Nama** | Fhaiz Rahmady Zaky |
| **NIM** | F1G124030 |

Prototype yang menerima citra ijazah dan menghasilkan:

```
Input  : ijazah_001.jpg
Output :
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
```

## Pipeline

```
Citra Ijazah → (koreksi orientasi) → Grayscale → Image Enhancement
      ┌────────────────────────────┴───────────────────────────┐
 Area Nomor Ijazah                                      Area Tanda Tangan
      ↓                                                         ↓
 Enhancement (8 metode dibandingkan)                      Thresholding
      ↓                                                         ↓
 OCR (Tesseract)                                           Morphology
      ↓                                                         ↓
 Nomor Ijazah                                         Signature Detection
      └────────────────────────────┬───────────────────────────┘
                          Hasil Verifikasi
```

Penjelasan metode lengkap ada di **[LAPORAN_METODE.md](LAPORAN_METODE.md)**.

## Struktur repository

```
ijazah-verifier/
├── main.py                    # CLI utama
├── evaluate_cer.py            # perbandingan metode enhancement berdasar CER
├── ijazah_verifier/
│   ├── preprocess.py          # grayscale, enhancement global, auto-orientasi
│   ├── enhancement.py         # 8 metode enhancement area nomor
│   ├── ocr_number.py          # lokalisasi area nomor + OCR
│   ├── signature.py           # thresholding + morphology + deteksi TTD
│   └── pipeline.py            # orkestrasi + penyimpanan citra debug
├── samples/                   # ijazah_001.jpg + ground_truth.csv
├── tests/test_pipeline.py
├── outputs/                   # contoh hasil (citra tiap tahap, tabel & grafik CER)
├── LAPORAN_METODE.md
└── requirements.txt
```

## How to run

### 1. Prasyarat
* Python 3.9+
* **Tesseract OCR** terpasang di sistem (bahasa `eng` sudah cukup, nomor hanya angka)

| OS | Perintah |
|---|---|
| Ubuntu/Debian | `sudo apt install tesseract-ocr` |
| macOS | `brew install tesseract` |
| Windows | Unduh installer UB Mannheim, lalu tambahkan folder instalasi ke `PATH` (atau set `pytesseract.pytesseract.tesseract_cmd`) |

### 2. Instalasi
```bash
git clone <URL_REPOSITORY_ANDA>
cd ijazah-verifier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib           # opsional, hanya untuk grafik CER
```

### 3. Menjalankan verifikasi
```bash
python main.py samples/ijazah_001.jpg
```
Keluaran:
```
Nomor Ijazah : 571012022000056
Tanda Tangan : PRESENT
Detail TTD   : {'Rektor': 'PRESENT', 'Dekan': 'PRESENT'}
Status       : VALID (nomor terbaca & tanda tangan ada)
```
Citra setiap tahap (grayscale, enhanced, ROI, threshold, morphology, dan citra
hasil beranotasi) disimpan di `outputs/<nama_file>/`.

Opsi tambahan:
```bash
python main.py ijazah.jpg --method 3_clahe   # pilih metode enhancement OCR
python main.py ijazah.jpg --json             # hasil lengkap format JSON
python main.py ijazah.jpg --no-debug         # tanpa menyimpan citra debug
```
Citra yang terputar 90°/180°/270° dikoreksi otomatis.

### 4. Evaluasi CER (metode enhancement terbaik)
```bash
python evaluate_cer.py                       # memakai samples/ground_truth.csv
```
Untuk menambah data uji: letakkan citra di `samples/` dan tambahkan baris
`nama_file.jpg,nomor_ijazah_sebenarnya` pada `samples/ground_truth.csv`.
Hasil: `outputs/cer/cer_table.md`, `cer_detail.csv`, `cer_chart.png`.

### 5. Test
```bash
python tests/test_pipeline.py      # atau: pip install pytest && pytest -q
```

## Hasil perbandingan enhancement (CER)

Ground truth: `571012022000056`. Setiap metode diuji pada 8 kondisi (1 bersih + 7 degradasi sintetis).

| Metode | clean | gaussian_noise | blur | low_contrast | uneven_shadow | salt_pepper | jpeg_q12 | combo(shadow+blur+noise) | **Rata-rata CER** |
|---|---|---|---|---|---|---|---|---|---|
| 8_kombinasi_full | 0.000 | 0.000 | 0.067 | 0.000 | 0.000 | 0.067 | 0.000 | 0.267 | **0.050** |
| 4_gaussian_otsu | 0.000 | 0.000 | 0.067 | 0.000 | 0.200 | 0.000 | 0.000 | 1.000 | **0.158** |
| 1_grayscale_saja | 0.000 | 0.000 | 0.000 | 0.000 | 0.067 | 1.000 | 0.000 | 1.000 | **0.258** |
| 7_bgnorm_otsu | 0.000 | 0.000 | 0.133 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | **0.267** |
| 6_unsharp_mask | 0.067 | 1.000 | 0.000 | 0.133 | 0.000 | 0.067 | 0.067 | 1.000 | **0.292** |
| 3_clahe | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.067 | 0.400 | **0.308** |
| 5_adaptive_threshold | 0.000 | 1.000 | 0.067 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | **0.383** |
| 2_histogram_equalization | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.133 | 0.133 | **0.658** |

**Metode terbaik (rata-rata CER terendah): `8_kombinasi_full` (0.050)**

Kesimpulan lengkap & keterbatasan ada di [LAPORAN_METODE.md](LAPORAN_METODE.md).

## Mengunggah ke GitHub
```bash
git init && git add . && git commit -m "Mini project verifikasi ijazah - Fhaiz Rahmady Zaky (F1G124030)"
git branch -M main
git remote add origin https://github.com/<username>/ijazah-verifier.git
git push -u origin main
```
Kumpulkan URL repository tersebut sebagai output tugas.

## Keterbatasan
* Lokasi area tanda tangan memakai rasio tetap sesuai **layout ijazah UI** (Rektor kiri, Dekan kanan). Template ijazah lain perlu menyesuaikan `SIGN_REGIONS` di `ijazah_verifier/signature.py`.
* Sistem hanya menilai **ada/tidaknya** goresan tanda tangan, bukan keaslian atau kecocokan pemilik.
* Evaluasi CER baru memakai 1 citra asli (+ degradasi sintetis); semakin banyak data semakin kuat kesimpulannya.
