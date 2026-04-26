"""
PoseCoach – Spor Hareket Analiz Araci
MediaPipe 0.10+ (Tasks API) ile uyumlu
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import argparse
import sys
import urllib.request
import os
import time
import threading
from collections import deque

# ─────────────────────────────────────────────
#  SES GERI BILDIRIMI (sadece Windows)
# ─────────────────────────────────────────────
try:
    import winsound
    _SOUND_OK = True
except ImportError:
    _SOUND_OK = False

def _beep(freq: int, dur: int):
    """Bloklama yapmadan ses cal."""
    if _SOUND_OK:
        threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()

def beep_rep():
    """Rep tamamlandiginda kisa onay sesi."""
    _beep(1050, 120)

def beep_warning():
    """Hatali form uyari sesi."""
    _beep(520, 280)

# ─────────────────────────────────────────────
#  MODEL INDIRME
# ─────────────────────────────────────────────

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[*] Model indiriliyor (ilk calistirma)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Model hazir.")

# ─────────────────────────────────────────────
#  LANDMARK INDEX
# ─────────────────────────────────────────────

LM = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
}

CONNECTIONS = [
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),   ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),     ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),         ("left_knee","left_ankle"),
    ("right_hip","right_knee"),       ("right_knee","right_ankle"),
]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def lm_xy(landmarks, name):
    lm = landmarks[LM[name]]
    return [lm.x, lm.y]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

class AngleSmoother:
    """Son N frame'in agirlikli ortalamasini alarak aci titresimini yok eder."""
    def __init__(self, window=8):
        self.buffer = deque(maxlen=window)

    def smooth(self, raw_angle):
        self.buffer.append(raw_angle)
        if len(self.buffer) == 1:
            return raw_angle
        weights = np.arange(1, len(self.buffer) + 1, dtype=float)
        weights /= weights.sum()
        return float(np.dot(list(self.buffer), weights))

def check_visibility(landmarks, names, threshold=0.55):
    """Belirtilen landmark'larin visibility skoru >= threshold mi kontrol eder."""
    for name in names:
        if landmarks[LM[name]].visibility < threshold:
            return False
    return True

# ─────────────────────────────────────────────
#  ANALYZERS
# ─────────────────────────────────────────────

class SquatAnalyzer:
    name = "Squat"
    required_landmarks = ["left_hip", "left_knee", "left_ankle", "left_shoulder"]
    angle_label      = "Diz Acisi"
    angle_thresholds = [95]

    def __init__(self):
        self.stage = None
        self.rep_count = 0
        self.angle_history = []
        self.knee_smoother = AngleSmoother(window=8)
        self.torso_smoother = AngleSmoother(window=8)

    def analyze(self, landmarks):
        hip      = lm_xy(landmarks, "left_hip")
        knee     = lm_xy(landmarks, "left_knee")
        ankle    = lm_xy(landmarks, "left_ankle")
        shoulder = lm_xy(landmarks, "left_shoulder")
        knee_angle  = self.knee_smoother.smooth(calculate_angle(hip, knee, ankle))
        torso_angle = self.torso_smoother.smooth(calculate_angle(shoulder, hip, knee))
        self.angle_history.append(round(knee_angle, 1))
        feedback, issues = [], []
        if knee_angle < 95:
            self.stage = "down"
        if knee_angle > 160 and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1
        if self.stage == "down":
            if knee_angle <= 95:
                feedback.append("[OK] Iyi derinlik!")
            else:
                issues.append("[v] Daha fazla comel")
        if abs(knee[0] - ankle[0]) > 0.07:
            issues.append("[!!] Dizler ice cokuyor!")
        else:
            feedback.append("[OK] Diz hizasi iyi")
        if torso_angle < 45:
            issues.append("[!!] Govde cok one egik")
        else:
            feedback.append("[OK] Sirt dik")
        return feedback + issues, len(issues) == 0, {"left_knee": knee_angle}

    def summary(self):
        good_frames = sum(1 for a in self.angle_history if a <= 95)
        pct = int(100 * good_frames / max(len(self.angle_history), 1))
        return [
            f"Toplam Rep      : {self.rep_count}",
            f"Iyi Form Suresi : %{pct}",
            f"Min Diz Acisi   : {min(self.angle_history, default=0):.1f} derece",
        ]


class PushupAnalyzer:
    name = "Push-up"
    required_landmarks = ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_ankle"]
    angle_label      = "Dirsek Acisi"
    angle_thresholds = [95]

    def __init__(self):
        self.stage = None
        self.rep_count = 0
        self.angle_history = []
        self.elbow_smoother = AngleSmoother(window=8)
        self.body_smoother = AngleSmoother(window=8)

    def analyze(self, landmarks):
        shoulder = lm_xy(landmarks, "left_shoulder")
        elbow    = lm_xy(landmarks, "left_elbow")
        wrist    = lm_xy(landmarks, "left_wrist")
        hip      = lm_xy(landmarks, "left_hip")
        ankle    = lm_xy(landmarks, "left_ankle")
        elbow_angle = self.elbow_smoother.smooth(calculate_angle(shoulder, elbow, wrist))
        body_angle  = self.body_smoother.smooth(calculate_angle(shoulder, hip, ankle))
        self.angle_history.append(round(elbow_angle, 1))
        feedback, issues = [], []
        if elbow_angle < 95:
            self.stage = "down"
        if elbow_angle > 155 and self.stage == "down":
            self.stage = "up"
            self.rep_count += 1
        if self.stage == "down":
            if elbow_angle <= 95:
                feedback.append("[OK] Yeterli derinlik!")
            else:
                issues.append("[v] Daha asagi in")
        if body_angle < 155:
            issues.append("[!!] Kalca dusuyor/yukseliyor")
        else:
            feedback.append("[OK] Vucut hatti duz")
        if abs(elbow[0] - shoulder[0]) > 0.10:
            issues.append("[!!] Dirsekler cok acik")
        else:
            feedback.append("[OK] Dirsek acisi iyi")
        return feedback + issues, len(issues) == 0, {"left_elbow": elbow_angle}

    def summary(self):
        good_frames = sum(1 for a in self.angle_history if a <= 95)
        pct = int(100 * good_frames / max(len(self.angle_history), 1))
        return [
            f"Toplam Rep      : {self.rep_count}",
            f"Iyi Form Suresi : %{pct}",
            f"Min Dirsek Acisi: {min(self.angle_history, default=0):.1f} derece",
        ]


class BicepCurlAnalyzer:
    name = "Bicep Curl"
    required_landmarks = ["right_shoulder", "right_elbow", "right_wrist",
                          "left_shoulder", "left_elbow", "left_wrist"]
    angle_label      = "Dirsek Acisi"
    angle_thresholds = [50, 150]

    def __init__(self):
        self.stage_r = None
        self.stage_l = None
        self.rep_count = 0
        self.angle_history = []
        self.elbow_smoother_r = AngleSmoother(window=8)
        self.elbow_smoother_l = AngleSmoother(window=8)

    def analyze(self, landmarks):
        r_shoulder = lm_xy(landmarks, "right_shoulder")
        r_elbow    = lm_xy(landmarks, "right_elbow")
        r_wrist    = lm_xy(landmarks, "right_wrist")
        
        l_shoulder = lm_xy(landmarks, "left_shoulder")
        l_elbow    = lm_xy(landmarks, "left_elbow")
        l_wrist    = lm_xy(landmarks, "left_wrist")
        
        r_angle = self.elbow_smoother_r.smooth(calculate_angle(r_shoulder, r_elbow, r_wrist))
        l_angle = self.elbow_smoother_l.smooth(calculate_angle(l_shoulder, l_elbow, l_wrist))
        
        active_angle = min(r_angle, l_angle)
        self.angle_history.append(round(active_angle, 1))
        
        feedback, issues = [], []
        
        # Sag kol kontrolu
        if r_angle > 150:
            self.stage_r = "down"
        if r_angle < 50 and self.stage_r == "down":
            self.stage_r = "up"
            self.rep_count += 1
            
        # Sol kol kontrolu
        if l_angle > 150:
            self.stage_l = "down"
        if l_angle < 50 and self.stage_l == "down":
            self.stage_l = "up"
            self.rep_count += 1
            
        # Geri bildirim icin aktif kola (daha cok bukulen kola) bakiyoruz
        active_stage = self.stage_r if r_angle < l_angle else self.stage_l
        if active_stage == "down":
            if active_angle >= 150:
                feedback.append("[OK] Tam eksantrik")
            else:
                issues.append("[v] Kolu tam ac (>150)")
        if active_stage == "up":
            if active_angle <= 50:
                feedback.append("[OK] Tam bukum")
            else:
                issues.append("[^] Daha fazla kiv (<50)")
                
        r_move = abs(r_elbow[1] - r_shoulder[1])
        l_move = abs(l_elbow[1] - l_shoulder[1])
        active_move = r_move if r_angle < l_angle else l_move
        if active_move < 0.05:
            issues.append("[!!] Dirsek ileri gidiyor!")
        else:
            feedback.append("[OK] Ust kol sabit")
            
        return feedback + issues, len(issues) == 0, {"right_elbow": r_angle, "left_elbow": l_angle}

    def summary(self):
        good_frames = sum(1 for a in self.angle_history if a <= 50 or a >= 150)
        pct = int(100 * good_frames / max(len(self.angle_history), 1))
        return [
            f"Toplam Rep      : {self.rep_count}",
            f"Iyi Form Suresi : %{pct}",
            f"Min Dirsek Acisi: {min(self.angle_history, default=0):.1f} derece",
            f"Max Dirsek Acisi: {max(self.angle_history, default=0):.1f} derece",
        ]

# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
C = {
    "good":    (0, 220, 100),
    "bad":     (40, 80, 255),
    "neutral": (180, 180, 180),
    "accent":  (0, 190, 255),
    "panel":   (20, 20, 35),
    "white":   (255, 255, 255),
}

def draw_skeleton(frame, landmarks, color, vid_h, vid_w, angles=None):
    """Koordinatlari SADECE video alani (vid_w x vid_h) icinde hesapla."""
    pts = {}
    for name, idx in LM.items():
        lm = landmarks[idx]
        # lm.x ve lm.y normalize [0,1] — sadece video parcasina gore scale et
        x = int(lm.x * vid_w)
        y = int(lm.y * vid_h)
        pts[name] = (x, y)
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame, pt, 7, color,        -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (255,255,255),  1, cv2.LINE_AA)
    # Eklem acilerini iskelet uzerine yaz
    if angles:
        for joint_name, angle_val in angles.items():
            if joint_name in pts:
                x, y = pts[joint_name]
                label = f"{angle_val:.0f}"
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.58, 2)
                cv2.rectangle(frame, (x + 8, y - th - 14), (x + 14 + tw, y - 6), (0, 0, 0), -1)
                cv2.putText(frame, label, (x + 10, y - 10), FONT, 0.58, color, 2, cv2.LINE_AA)

def draw_panel(frame, feedback, is_good, ex_name, reps, history, vid_w,
               angle_history=None, angle_label="", angle_thresholds=None):
    h, w = frame.shape[:2]
    x0 = vid_w + 12

    # Panel arka plani
    overlay = frame.copy()
    cv2.rectangle(overlay, (vid_w, 0), (w, h), C["panel"], -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # Ust bilgiler
    cv2.putText(frame, ex_name.upper(), (x0, 45),  FONT, 0.72, C["accent"], 2, cv2.LINE_AA)
    rep_col = C["good"] if is_good else C["bad"]
    cv2.putText(frame, f"REPS: {reps}",  (x0, 88),  FONT, 1.0,  rep_col, 3, cv2.LINE_AA)
    status  = "DOGRU FORM" if is_good else "DUZELT"
    cv2.putText(frame, status,           (x0, 126), FONT, 0.62, rep_col, 2, cv2.LINE_AA)

    # Feedback satirlari (max 6 satir)
    y = 164
    for line in feedback[:6]:
        col = C["good"] if line.startswith("[OK]") else C["bad"]
        cv2.putText(frame, line, (x0, y), FONT, 0.50, col, 1, cv2.LINE_AA)
        y += 24

    # ── Gercek zamanli aci grafigi ──────────────────────────
    if angle_history and len(angle_history) > 1:
        gx = x0
        gw = w - vid_w - 20
        gt = 320          # grafik ust kenari
        gh = 85           # grafik yuksekligi (piksel)
        gb = gt + gh      # grafik alt kenari

        # Arka plan kutusu
        cv2.rectangle(frame, (gx - 2, gt - 22), (gx + gw, gb + 6), (28, 28, 46), -1)
        cv2.rectangle(frame, (gx - 2, gt - 22), (gx + gw, gb + 6), C["neutral"],  1)

        # Esik cizgileri (acik mavi)
        if angle_thresholds:
            for thresh in angle_thresholds:
                ty = gb - int((thresh / 180.0) * gh)
                ty = max(gt, min(gb, ty))
                cv2.line(frame, (gx, ty), (gx + gw - 4, ty), (40, 210, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{int(thresh)}",
                            (gx + gw - 34, ty - 3), FONT, 0.36, (40, 210, 255), 1, cv2.LINE_AA)

        # Cizgi grafigi
        subset = list(angle_history)[-(gw):]
        n = len(subset)
        step = (gw - 4) / max(n - 1, 1)
        pts = []
        for i, val in enumerate(subset):
            px = gx + int(i * step)
            py = gb - int((float(val) / 180.0) * gh)
            py = max(gt, min(gb, py))
            pts.append([px, py])
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_arr], False, C["accent"], 2, cv2.LINE_AA)

        # Son deger etiketi (sag ust)
        last_v = list(angle_history)[-1]
        cv2.putText(frame, f"{last_v:.1f}",
                    (gx + gw - 54, gt - 24), FONT, 0.42, C["accent"], 1, cv2.LINE_AA)
        # Grafik basligi (sol ust)
        cv2.putText(frame, angle_label if angle_label else "Aci",
                    (gx, gt - 24), FONT, 0.40, C["neutral"], 1, cv2.LINE_AA)

    # Form gecmisi barlari
    if history:
        bx   = x0
        by   = h - 85
        pw   = w - vid_w - 20
        bw   = max(3, pw // max(len(history), 1))
        for i, g in enumerate(history):
            cv2.rectangle(frame,
                (bx + i*bw, by - 25), (bx + i*bw + bw - 2, by),
                C["good"] if g else C["bad"], -1)
        cv2.putText(frame, "Form Gecmisi", (bx, by + 15),
                    FONT, 0.42, C["neutral"], 1, cv2.LINE_AA)

def show_summary_screen(analyzer, screen_w, screen_h):
    """Video bittikten sonra ozet ekrani goster."""
    sw = min(640, screen_w)
    sh = min(400, screen_h)
    summary_frame = np.zeros((sh, sw, 3), dtype=np.uint8)
    summary_frame[:] = (20, 20, 35)

    lines = analyzer.summary()
    total_good = sum(1 for l in lines)  # just a count placeholder

    cv2.putText(summary_frame, "ANALIZ TAMAMLANDI", (30, 55),
                FONT, 0.9, C["accent"], 2, cv2.LINE_AA)
    cv2.putText(summary_frame, analyzer.name.upper(), (30, 95),
                FONT, 0.7, C["white"], 1, cv2.LINE_AA)
    cv2.line(summary_frame, (30, 110), (sw - 30, 110), C["neutral"], 1)

    y = 150
    for line in lines:
        cv2.putText(summary_frame, line, (50, y), FONT, 0.65, C["good"], 1, cv2.LINE_AA)
        y += 40

    cv2.putText(summary_frame, "Kapatmak icin herhangi bir tusa basin...",
                (30, sh - 25), FONT, 0.45, C["neutral"], 1, cv2.LINE_AA)

    cv2.imshow("PoseCoach — Ozet", summary_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─────────────────────────────────────────────
#  PDF RAPOR
# ─────────────────────────────────────────────

def generate_pdf_report(analyzer, source_label, duration_s):
    """Analiz bittikten sonra PDF rapor olustur."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile, datetime
    except ImportError as e:
        print(f"\n[!] PDF icin eksik paket: {e}")
        print("    pip install reportlab matplotlib")
        return None

    # ─ Dosya adi
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"PoseCoach_{analyzer.name.replace(' ', '_')}_{ts}.pdf"
    now_str  = datetime.datetime.now().strftime("%d.%m.%Y  %H:%M")
    mins, secs = divmod(int(duration_s), 60)

    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm,   bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # ─ Stil tanimlari
    s_title = ParagraphStyle("pc_title", parent=styles["Heading1"],
                             fontSize=20, textColor=colors.HexColor("#00BEFF"),
                             spaceAfter=4)
    s_sub   = ParagraphStyle("pc_sub",   parent=styles["Normal"],
                             fontSize=10, textColor=colors.HexColor("#888888"),
                             spaceAfter=10)
    s_h2    = ParagraphStyle("pc_h2",    parent=styles["Heading2"],
                             fontSize=13, textColor=colors.HexColor("#14141E"),
                             spaceBefore=14, spaceAfter=6)
    s_foot  = ParagraphStyle("pc_foot",  parent=styles["Normal"],
                             fontSize=8,  textColor=colors.HexColor("#AAAAAA"),
                             alignment=TA_CENTER)
    s_cell  = styles["Normal"]

    ACCENT  = colors.HexColor("#00BEFF")
    DARK    = colors.HexColor("#14141E")
    LIGHT   = colors.HexColor("#F4F9FF")

    story = []

    # ─ Baslik
    story.append(Paragraph("PoseCoach — Analiz Raporu", s_title))
    story.append(Paragraph(
        f"Egzersiz: <b>{analyzer.name}</b> &nbsp;&nbsp;|"
        f"&nbsp;&nbsp;Tarih: {now_str} &nbsp;&nbsp;|"
        f"&nbsp;&nbsp;Kaynak: {source_label} &nbsp;&nbsp;|"
        f"&nbsp;&nbsp;Sure: {mins} dk {secs} sn", s_sub))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=ACCENT, spaceAfter=10))

    # ─ Ozet istatistikler
    story.append(Paragraph("Genel Sonuclar", s_h2))
    summary_lines = analyzer.summary()
    tbl_data = [[Paragraph("<b>" + l.split(":")[0].strip() + "</b>", s_cell),
                 Paragraph(l.split(":", 1)[1].strip() if ":" in l else "", s_cell)]
                for l in summary_lines]
    tbl = Table(tbl_data, colWidths=[9*cm, 7*cm])
    tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("FONTSIZE",      (0, 0), (-1, -1), 11),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.4*cm))

    # ─ Istatistik tablosu
    hist = analyzer.angle_history
    if hist:
        story.append(Paragraph("Detayli Istatistikler", s_h2))
        stat_rows = [
            [Paragraph("<b>Metrik</b>", s_cell),
             Paragraph("<b>Deger</b>",  s_cell)],
            [Paragraph("Ortalama Aci",  s_cell),
             Paragraph(f"{sum(hist)/len(hist):.1f} derece", s_cell)],
            [Paragraph("Minimum Aci",   s_cell),
             Paragraph(f"{min(hist):.1f} derece", s_cell)],
            [Paragraph("Maximum Aci",   s_cell),
             Paragraph(f"{max(hist):.1f} derece", s_cell)],
            [Paragraph("Analiz Edilen Frame", s_cell),
             Paragraph(str(len(hist)), s_cell)],
        ]
        stat_tbl = Table(stat_rows, colWidths=[9*cm, 7*cm])
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT, colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("FONTSIZE",      (0, 0), (-1, -1), 11),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 0.5*cm))

    # ─ Aci grafigi (matplotlib)
    tmp_path = None
    if hist and len(hist) > 1:
        story.append(Paragraph(f"{analyzer.angle_label} Grafigi", s_h2))

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.set_facecolor("#1A1A2E")
        fig.patch.set_facecolor("#1A1A2E")

        ax.plot(range(len(hist)), hist,
                color="#00BEFF", linewidth=1.5, label=analyzer.angle_label)
        for thresh in (analyzer.angle_thresholds or []):
            ax.axhline(y=thresh, color="#FFD700", linestyle="--",
                       linewidth=1.2, label=f"Esik: {thresh}\u00b0")

        ax.set_xlabel("Frame",        color="white", fontsize=10)
        ax.set_ylabel("Aci (derece)", color="white", fontsize=10)
        ax.set_title(f"{analyzer.name} — {analyzer.angle_label}",
                     color="white", fontsize=11)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.legend(facecolor="#333333", labelcolor="white", fontsize=9)
        ax.set_ylim(0, 185)
        plt.tight_layout()

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(tmp_fd)
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight",
                    facecolor="#1A1A2E")
        plt.close(fig)

        story.append(RLImage(tmp_path, width=16*cm, height=5*cm))
        story.append(Spacer(1, 0.3*cm))

    # ─ Footer
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#CCCCCC"), spaceAfter=6))
    story.append(Paragraph(
        "Bu rapor PoseCoach tarafindan otomatik olusturulmustur.", s_foot))

    doc.build(story)
    
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
            
    print(f"[PDF] Rapor kaydedildi: {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
#  LANDMARKER FACTORY
# ─────────────────────────────────────────────

def make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run(video_path, exercise: str):
    download_model()

    analyzers = {"squat": SquatAnalyzer(), "pushup": PushupAnalyzer(), "curl": BicepCurlAnalyzer()}
    analyzer  = analyzers[exercise]

    is_camera = str(video_path) == "0"
    cap = cv2.VideoCapture(0 if is_camera else video_path)
    if not cap.isOpened():
        src_label = "Canli Kamera" if is_camera else video_path
        print(f"[HATA] Kaynak acilamadi: {src_label}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_ms = int(1000 / fps)

    # Ekran cozunurlugunu al
    screen_w, screen_h = 1280, 720
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        pass

    PANEL_W   = 310
    max_vid_w = screen_w - PANEL_W - 40
    max_vid_h = screen_h - 60

    landmarker   = make_landmarker()
    history      = deque(maxlen=60)
    bad_frame_count    = 0       # Hysteresis: ardisik hatali frame sayaci
    BAD_FRAME_THRESH   = 8       # Kac frame ust uste hatali olursa goster
    timestamp_ms     = 1
    prev_rep_count   = 0
    last_warning_t   = 0.0
    WARNING_COOLDOWN = 2.0       # saniye — uyari arasi minimum sure
    current_angles   = {}

    kaynak = "Canli Kamera" if is_camera else os.path.basename(str(video_path))
    t_start = time.time()
    print(f"\n\u25b6  {analyzer.name} analizi basliyor... [{kaynak}]  (Q veya ESC ile cikis)\n")

    while cap.isOpened():
        ret, frame = cap.read()

        # Video bitti — dongu yok, dur
        if not ret:
            break

        # Video karesini ekrana sigidir (en-boy oranini koru)
        h0, w0 = frame.shape[:2]
        scale  = min(max_vid_w / w0, max_vid_h / h0)
        vid_w  = int(w0 * scale)
        vid_h  = int(h0 * scale)
        vid_frame = cv2.resize(frame, (vid_w, vid_h))

        # Pose tespiti SADECE video karesi uzerinde yap
        if is_camera:
            timestamp_ms = int(time.time() * 1000)
        else:
            timestamp_ms += frame_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Saga panel ekle (pose tespitinden SONRA)
        canvas = np.zeros((vid_h, vid_w + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :vid_w] = vid_frame
        frame = canvas

        feedback, is_good = ["Kisi algilanamadi..."], False
        current_angles = {}

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # 1) Visibility filtresi
            if not check_visibility(landmarks, analyzer.required_landmarks):
                feedback = ["[!] Vucut gorunmuyor..."]
                is_good  = False
                bad_frame_count = 0
            else:
                # 2) Analiz (smoothed acilarla)
                raw_feedback, raw_good, current_angles = analyzer.analyze(landmarks)

                # 3) Hysteresis gate
                if raw_good:
                    bad_frame_count = 0
                    feedback, is_good = raw_feedback, True
                else:
                    bad_frame_count += 1
                    if bad_frame_count >= BAD_FRAME_THRESH:
                        feedback, is_good = raw_feedback, False
                    else:
                        # Esik asilmadi — hatalari gizle
                        feedback = [f for f in raw_feedback if f.startswith("[OK]")]
                        is_good = True

            # Ses geri bildirimi
            if analyzer.rep_count > prev_rep_count:
                beep_rep()
                prev_rep_count = analyzer.rep_count
            now = time.time()
            if not is_good and (now - last_warning_t) >= WARNING_COOLDOWN:
                beep_warning()
                last_warning_t = now

            history.append(is_good)
            # Iskeleti video alani boyutlariyla ciz (panel offset yok)
            draw_skeleton(frame, landmarks, C["good"] if is_good else C["bad"], vid_h, vid_w, current_angles)

        draw_panel(frame, feedback, is_good, analyzer.name, analyzer.rep_count, list(history), vid_w,
                   list(analyzer.angle_history), analyzer.angle_label, analyzer.angle_thresholds)

        cv2.imshow(f"PoseCoach — {analyzer.name}", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Ozet ekrani goster
    print(f"\n{'='*40}")
    print(f"  ANALIZ TAMAMLANDI — {analyzer.name}")
    print(f"{'='*40}")
    for line in analyzer.summary():
        print(f"  {line}")
    print(f"{'='*40}\n")

    show_summary_screen(analyzer, screen_w, screen_h)
    generate_pdf_report(analyzer, kaynak, time.time() - t_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseCoach")
    parser.add_argument("video",    help="Video dosyasi")
    parser.add_argument("exercise", choices=["squat", "pushup", "curl"])
    args = parser.parse_args()
    run(args.video, args.exercise)
