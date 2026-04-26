"""
PoseCoach Launcher
------------------
Dosya seçmek için GUI dialog açar, hareket seçtikten sonra analizi başlatır.
Bağımlılık: pip install mediapipe opencv-python numpy
"""

import subprocess
import sys
import os

def check_deps():
    missing = []
    for pkg in ["cv2", "mediapipe", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        real_names = {"cv2": "opencv-python"}
        install = [real_names.get(p, p) for p in missing]
        print(f"[!] Eksik paketler: {missing}")
        print(f"    Kuruluyor: pip install {' '.join(install)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + install)


def pick_source():
    """Video dosyasi sec veya canli kamera kullan."""
    print("\nKaynak sec:")
    print("  1. Video dosyasi sec")
    print("  2. Canli kamera (webcam)")
    while True:
        ch = input("Numara gir (1/2): ").strip()
        if ch == "2":
            return "0"          # kamera sentinel
        elif ch == "1":
            return pick_file()
        print("  Gecersiz secim, tekrar dene.")


def pick_file():
    """Try tkinter dialog, fall back to manual input."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Video Sec",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Tumu", "*.*")]
        )
        root.destroy()
        return path or None
    except Exception:
        return input("Video dosyasi yolu: ").strip()


def pick_exercise():
    print("\nEgzersiz seç:")
    options = {"1": ("squat",  "Squat (Çömelme)"),
               "2": ("pushup", "Push-up (Şınav)"),
               "3": ("curl",   "Bicep Curl")}
    for k, (_, label) in options.items():
        print(f"  {k}. {label}")
    while True:
        choice = input("\nNumara gir (1/2/3): ").strip()
        if choice in options:
            return options[choice][0]
        print("  Geçersiz seçim, tekrar dene.")


if __name__ == "__main__":
    print("=" * 50)
    print("  PoseCoach – Spor Hareket Analiz Aracı")
    print("=" * 50)

    check_deps()

    source = pick_source()
    if source != "0" and (not source or not os.path.isfile(source)):
        print("[!] Gecerli bir video secilmedi.")
        sys.exit(1)

    exercise = pick_exercise()

    script = os.path.join(os.path.dirname(__file__), "pose_coach.py")
    label  = "Canli Kamera" if source == "0" else os.path.basename(source)
    print(f"\n\u25b6 Baslatiliyor: {exercise} \u2192 {label}\n")
    subprocess.run([sys.executable, script, source, exercise])
