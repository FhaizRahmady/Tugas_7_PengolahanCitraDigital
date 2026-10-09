"""CLI: python main.py samples/ijazah_001.jpg"""
import argparse
import json
import sys
from ijazah_verifier.pipeline import verify
from ijazah_verifier.enhancement import METHODS, DEFAULT_METHOD


def main():
    ap = argparse.ArgumentParser(description="Verifikasi ijazah: OCR nomor + deteksi tanda tangan")
    ap.add_argument("image", help="path citra ijazah (jpg/png)")
    ap.add_argument("--out", default="outputs", help="folder output debug (default: outputs)")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=list(METHODS),
                    help="metode enhancement untuk OCR")
    ap.add_argument("--json", action="store_true", help="cetak hasil lengkap sebagai JSON")
    ap.add_argument("--no-debug", action="store_true", help="jangan simpan citra debug")
    a = ap.parse_args()

    r = verify(a.image, a.out, a.method, save_debug=not a.no_debug)
    print(f"Input  : {a.image}")
    print(f"Nomor Ijazah : {r['nomor_ijazah']}")
    print(f"Tanda Tangan : {r['tanda_tangan']}")
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print(f"Detail TTD   : {r['detail_ttd']}")
        print(f"Status       : {r['status']}")
        if not a.no_debug:
            print(f"Citra debug  : {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
