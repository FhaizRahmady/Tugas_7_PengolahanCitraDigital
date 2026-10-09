# Laporan Metode — Verifikasi Ijazah

**Nama:** Fhaiz Rahmady Zaky  **NIM:** F1G124030

## 1. Tujuan
Membaca nomor ijazah dengan OCR dan menentukan ada/tidaknya tanda tangan pimpinan (Rektor/Dekan; pada ijazah perguruan tinggi setara "kepala sekolah") menggunakan teknik pengolahan citra digital.

## 2. Metode per tahap

**0. Koreksi orientasi.** Scan sering terputar 90°. Sistem mencoba rotasi 0/90/180/270° (hanya yang landscape), menjalankan OCR cepat pada citra yang diperkecil, dan memilih rotasi yang menghasilkan kata kunci ijazah terbanyak ("universitas", "ijazah", "rektor", dst).

**1. Grayscale.** Konversi BGR → intensitas (`cv2.cvtColor`). Informasi warna (kertas kuning, stempel biru) tidak diperlukan untuk OCR maupun deteksi goresan, dan menyederhanakan proses selanjutnya.

**2. Image Enhancement global.** Kertas ijazah memiliki pola guilloche dan warna latar yang tidak rata.
* *Normalisasi latar*: citra dibagi estimasi latar (median blur 51×51) → latar menjadi ≈putih merata, teks/tinta tetap gelap.
* *CLAHE* (clip 2.0, tile 8×8): menaikkan kontras lokal tanpa memperbesar noise berlebihan.

**3a. Cabang Nomor Ijazah**
1. *Lokalisasi area*: Tesseract mencari kata "Nomor" di 30% bagian bawah-kiri; ROI satu baris dibentuk dari posisinya. Jika gagal dipakai ROI cadangan (rasio tetap).
2. *Enhancement area nomor*: ROI di-upscale 2× (bicubic), lalu diproses salah satu dari 8 metode (bagian 4). Metode bawaan: **kombinasi**.
3. *OCR*: Tesseract `--psm 7` (satu baris). Bagian kanan label "ijazah:" kemudian dipotong dan di-OCR ulang dengan *whitelist* digit `0-9-`, hasil dipilih lewat regex deret angka terpanjang.

**3b. Cabang Tanda Tangan**
1. *Area*: ROI Rektor dan Dekan (rasio layout), dipilih di antara teks cetak "Rektor/Dekan" dan nama pejabat agar teks cetak tidak ikut.
2. *Thresholding*: normalisasi latar → Gaussian blur → ambang `min(Otsu, 150)`. Batas absolut 150 penting: pada ROI kosong Otsu tetap membelah pola latar sehingga muncul "tinta" palsu (false positive terbukti pada uji negatif, lalu diperbaiki).
3. *Morfologi*: opening ellipse 3×3 (buang noise latar) → closing ellipse 15×15 (menyambung goresan tanda tangan yang tipis/putus).
4. *Signature Detection*: connected components pada hasil closing. Sebuah komponen dianggap tanda tangan bila lebar ≥ 7% lebar citra, jumlah piksel tinta asli ≥ 0,4% luas ROI, dan bentuknya memanjang/tinggi (bukan titik/teks kecil). Hasil akhir **PRESENT** bila minimal satu area (Rektor/Dekan) terdeteksi, selain itu **ABSENT**.

**4. Hasil verifikasi.** Nomor + status tanda tangan; status "VALID" bila nomor terbaca dan tanda tangan ada, selain itu "PERLU DICEK MANUAL". Citra beranotasi disimpan.

## 3. Metode enhancement yang dibandingkan (area nomor)
| # | Metode | Ringkas |
|---|---|---|
| 1 | Grayscale saja | pembanding (baseline) |
| 2 | Histogram Equalization | pemerataan histogram global |
| 3 | CLAHE | kontras adaptif lokal |
| 4 | Gaussian blur + Otsu | thresholding global |
| 5 | Adaptive threshold (Gaussian) | threshold lokal |
| 6 | Unsharp mask | penajaman tepi |
| 7 | Normalisasi latar + Otsu | koreksi pencahayaan + threshold |
| 8 | **Kombinasi** | median 3×3 → normalisasi latar → Gaussian σ1.5 → Otsu → *despeckle* (buang komponen < 30 px) → closing 2×2 |

## 4. Evaluasi CER
`CER = (S + D + I) / N` dengan jarak Levenshtein antara hasil OCR dan ground truth `571012022000056` (N = 15). Karena scan asli sudah bersih (semua metode umumnya CER 0), ROI nomor diuji juga pada 7 degradasi sintetis (seed tetap = 42, dapat diulang): noise Gaussian, blur, kontras rendah, bayangan tak merata, salt & pepper, kompresi JPEG kualitas 12, dan kombinasi bayangan+blur+noise.

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

## 5. Analisis: metode paling efektif
**Metode 8 (kombinasi)** memberi rata-rata CER terendah. Alasannya:
* *Normalisasi latar* menghilangkan bayangan/gradasi pencahayaan (CER 0 pada kondisi bayangan, sedangkan Gaussian+Otsu murni gagal 0,200).
* *Median + despeckle* membuang noise impulsif dan bintik pola guilloche yang lolos threshold (salt & pepper dan noise: hampir tanpa kesalahan, sedangkan threshold tanpa pembersihan kegagalan total).
* Threshold menghasilkan teks hitam-putih bersih yang disukai Tesseract.

Metode lain: *Histogram Equalization* paling buruk karena memperkuat pola guilloche latar sama kuatnya dengan teks; *CLAHE* gagal pada citra kontras rendah (memperkuat noise); *adaptive threshold* sangat sensitif noise; *unsharp mask* memperkuat noise sehingga gagal pada citra berderau.

## 6. Jujur soal keterbatasan evaluasi
* Hanya 1 citra asli dan 15 karakter ground truth, sehingga satu karakter salah = CER 0,067; angka rata-rata bersifat indikatif, bukan kesimpulan statistik.
* Degradasi bersifat sintetis. Parameter metode 8 (mis. ukuran despeckle) disetel setelah melihat kegagalan pada uji ini, sehingga keunggulannya cenderung optimistis. Pengujian pada lebih banyak ijazah asli diperlukan.
* Deteksi tanda tangan memakai ROI berbasis layout; diuji pada 1 sampel positif dan 1 sampel negatif sintetis (tanda tangan diganti kertas kosong), plus uji rotasi.
