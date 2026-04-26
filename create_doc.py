import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem

def create_project_doc():
    pdf_path = "PoseCoach_Proje_Dokuman.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=24, textColor=colors.HexColor("#1A1A2E"),
        spaceAfter=12, alignment=1 # Center
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading2"],
        fontSize=16, textColor=colors.HexColor("#0F3460"),
        spaceBefore=18, spaceAfter=8,
        borderPadding=4,
        backColor=colors.HexColor("#F4F9FF")
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading3"],
        fontSize=13, textColor=colors.HexColor("#1A1A2E"),
        spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"
    )
    p_style = ParagraphStyle(
        "P", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#333333"),
        spaceAfter=8, leading=16
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#333333"),
        spaceAfter=4, leading=16, leftIndent=15
    )

    story = []
    
    # Title
    story.append(Paragraph("PoseCoach Teknik Proje Dosyasi", title_style))
    story.append(Paragraph("Mimarî, Kararlar ve Algoritmik Yontemler", ParagraphStyle("sub", parent=p_style, alignment=1, textColor=colors.gray)))
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#E94560"), spaceAfter=15))

    # Section 1
    story.append(Paragraph("1. Projenin Amaci ve Kapsami", h1_style))
    story.append(Paragraph(
        "PoseCoach, bilgisayarli goru (computer vision) ve makine ogrenmesi tekniklerini kullanarak sporcularin veya fitness "
        "meraklilarinin egzersiz formlarini gercek zamanli analiz eden bir masaustu asistanidir. Amaci, yanlis hareket formundan "
        "kaynaklanan sakatliklari onlemek ve kas gelisimini optimize etmektir. Proje; Squat, Push-up ve Bicep Curl hareketlerini destekler.", 
        p_style
    ))

    # Section 2
    story.append(Paragraph("2. Kullanilan Teknolojiler ve Modeller", h1_style))
    
    story.append(Paragraph("Neden MediaPipe Pose Landmarker Lite Secildi?", h2_style))
    story.append(Paragraph(
        "Piyasada YOLOv8-Pose gibi cok guclu poz tespiti modelleri bulunmasina ragmen, bu projede bilincli olarak Google'in "
        "<b>MediaPipe Tasks API (Pose Landmarker Lite)</b> modeli tercih edilmistir. Bunun baslica nedenleri:", p_style
    ))
    story.append(Paragraph("<b>• Performans ve Hiz Dengesi:</b> Lite modeli, ozel bir ekran kartina (GPU) ihtiyac duymadan, standart islemciler (CPU) uzerinde 30 FPS ve uzeri gercek zamanli akici analiz yapabilir. (XNNPACK CPU Delegasyonu kullanilmistir).", bullet_style))
    story.append(Paragraph("<b>• Hafiflik:</b> Modelin cok kucuk boyutu (~5MB) sayesinde sisteme aninda yuklenebilir ve bellegi yormaz.", bullet_style))
    story.append(Paragraph("<b>• Topolojik Uyum:</b> Vucuttaki 33 kritik eklem noktasini (landmark) normalize edilmis (0-1 arasi) 3D uzay koordinatlari ile vererek aci hesaplamalarini cok kolaylastirir.", bullet_style))

    story.append(Paragraph("Neden Python ve OpenCV?", h2_style))
    story.append(Paragraph(
        "<b>OpenCV (cv2)</b>, matris hesaplamalari (Numpy entegrasyonu) ve ekrana gercek zamanli cizim yapabilme gucu (iskelet "
        "cizimi, UI paneli, dinamik grafikler) icin kullanildi. Arayuz icin ayri bir kutuphane (PyQt/Tkinter) kullanmak yerine "
        "dogrudan video frame'i uzerine bir HUD (Head-up Display) kodlanarak gecikme sifira indirildi.", p_style
    ))

    # Section 3
    story.append(Paragraph("3. Uygulanan Matematiksel ve Algoritmik Yontemler", h1_style))
    
    story.append(Paragraph("A) Vektorel Aci Hesaplamasi (Cosine Rule)", h2_style))
    story.append(Paragraph(
        "Eklem acilari, uzaydaki uc nokta (A, B, C) kullanilarak hesaplanmistir. Matematiksel olarak, merkez nokta B alinarak "
        "BA ve BC vektorleri olusturulmus, Numpy <b>Dot Product (Nokta Carpimi)</b> kullanilarak aralarindaki kosinus degeri "
        "bulunmus ve ters kosinus (arccos) ile dereceye cevrilmistir.", p_style
    ))

    story.append(Paragraph("B) Sinyal Isleme: Agirlikli Hareketli Ortalama (Weighted Moving Average)", h2_style))
    story.append(Paragraph(
        "Makine ogrenmesi modelleri aydinlatma ve hareket bulanikligi (motion blur) sebebiyle titreyen (jitter) sonuclar verebilir. "
        "Bunu cozmek icin <b>AngleSmoother</b> isimli ozel bir sinif kodlanmistir. Bu sinif, son N frame'in agirlikli ortalamasini "
        "alarak (yeni framelere daha yuksek agirlik vererek) aci grafigindeki ani ziplamalari tiraslar ve puruzsuz bir veri akisi saglar.", p_style
    ))

    story.append(Paragraph("C) Hysteresis (Gecikmeli Durum Gecisi) Yontemi", h2_style))
    story.append(Paragraph(
        "Sistemin saliselik yanlis tespitlerde kullaniciya hemen 'Hatali Form' uyarisi verip kafa karistirmamasi icin Hysteresis kalkani "
        "kurulmustur. Hata tespit edildiginde sistem bir sayac baslatir (bad_frame_count). Sadece art arda belirlenen esik degerini "
        "(BAD_FRAME_THRESH=8) asan hatalar ekrana ve sese yansitilir.", p_style
    ))

    story.append(Paragraph("D) Durum Makinesi (State Machine) Mantigi", h2_style))
    story.append(Paragraph(
        "Her analizorde bir 'stage' (durum) degiskeni bulunur ('up' veya 'down'). Hareket eksantrik ve konsantrik fazlara ayrilarak "
        "kullanicinin tam hareket araligini (Full Range of Motion) tamamlayip tamamlamadigi kontrol edilir ve tekrarlar sadece "
        "iki faz basariyla tamamlandiginda sayilir.", p_style
    ))

    # Section 4
    story.append(Paragraph("4. Egzersiz Analizi Mantiklari", h1_style))
    story.append(Paragraph("<b>• Squat:</b> Kalca-Diz-Bilek acisina bakarak inme derinligi (<95°) tespit edilir. Omuz-Kalca-Diz acisi ile sirtin dikligi kontrol edilir. Ayrica ayak bilegi ve diz x-koordinatlari karsilastirilarak 'dizlerin ice cokme' hatasi yakalanir.", bullet_style))
    story.append(Paragraph("<b>• Push-up (Sinav):</b> Dirsek acisi ile derinlik olculur. Omuz-Kalca-Ayak bilegi acisi ile vucut formunun duz olup olmadigi (<155°) denetlenir. Ayrica dirsegin omuzdan disari cok acilip acilmadigi takip edilir.", bullet_style))
    story.append(Paragraph("<b>• Bicep Curl:</b> Her iki kol da ayni anda takip edilir (Aktif kol tespiti). Kollari tam acip acmadigi (>150°) ve tam bukup bukmedigi (<50°) denetlenirken, omuz ile dirsegin y-koordinati karsilastirilarak dirsegin sabit kalip kalmadigi kontrol edilir.", bullet_style))

    # Section 5
    story.append(Paragraph("5. Gelismis Kullanici Ozellikleri", h1_style))
    story.append(Paragraph("<b>• Canli Aci Grafigi (Polyline Plotting):</b> Matplotlib gibi agir kutuphaneler yerine, performans icin OpenCV'nin polylines fonksiyonuyla sifirdan bir line chart cizilmis ve limit cizgileriyle gorsellestirilmistir.", bullet_style))
    story.append(Paragraph("<b>• Asenkron Sesli Geri Bildirim:</b> Goruntu akisinin dondurulmamasi icin Windows'un winsound kutuphanesi <b>Threading (Multi-threading)</b> ile asenkron olarak kullanilmis, rep sayiminda ve hatalarda kullaniciya aninda tepki verilmistir.", bullet_style))
    story.append(Paragraph("<b>• Dinamik PDF Raporlama:</b> Program kapandiginda, toplanan aci history listesi ReportLab ve Matplotlib kullanilarak analiz edilir. Hata yuzdeleri ve performans grafigi PDF formatinda profesyonel bir cikti olarak sunulur.", bullet_style))

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Sonuc", h2_style))
    story.append(Paragraph(
        "PoseCoach, guncel yapay zeka yaklasimlariyla (Pose Estimation), klasik yazilim algoritmalarinin (Sinyal isleme, Hysteresis) "
        "harmanlandigi, performansi yuksek ve sunuma/ticari urune donusmeye hazir bir prototiptir.", p_style
    ))

    doc.build(story)
    print(f"Rapor olusturuldu: {pdf_path}")

if __name__ == '__main__':
    create_project_doc()
