# WRC Rally1 Hybrid Simulator

C++17 ve OpenGL (Fixed-Function Pipeline / X11) ile sıfırdan geliştirilmiş, hiçbir harici oyun motoru (Unity/Unreal vb.) gerektirmeyen tek dosyalı WRC Rally1 Hybrid yarış simülatörü.

## Özellikler
- **3 Resmi WRC Rally1 Aracı:**
  - Toyota GR Yaris Rally1 Hybrid (#18 S. Ogier)
  - Hyundai i20 N Rally1 Hybrid (#11 T. Neuville)
  - M-Sport Ford Puma Rally1 Hybrid (#16 A. Fourmaux)
- **Detaylı 3D Araç Modeli:** Swan-neck çift kademeli arka kanat, geniş gövde (widebody) çamurluklar, 3D LED far & stop barları, arka cam rezistans telleri, 3. fren lambası, FIA yarış kapı numaralıkları, güneşlik (sunstrip) ve tam takla kafesli kokpit içi göstergeler (MoTeC dijital telemetri, el freni, tripmeter).
- **Gerçekçi Fizik & Hibrit Güç Sistemi:** 240 Hz alt-adımlı (sub-stepping) araç dinamiği, süspansiyon esnemesi, ağırlık transferi, kayma açısı (slip angle), 4 tekerlek lastik izi/toz partikülleri ve hibrit boost sistemi.
- **2D Serbest Çizim Harita Tasarımcısı:** Kendi etap rotanızı fareyle çizip 3D dünyaya Catmull-Rom spline enterpolasyonu ile dönüştürme.
- **Dinamik Hava/İklim:** Güneşli toprak, asfalt, yağmur ve kar koşulları.

## Gereksinimler (Linux)
- C++17 destekli derleyici (`g++` veya `clang++`)
- X11 & OpenGL kütüphaneleri: `libX11-devel`, `mesa-libGL-devel`, `mesa-libGLU-devel` (veya Debian/Ubuntu: `libx11-dev`, `libgl1-mesa-dev`, `libglu1-mesa-dev`)

## Derleme ve Çalıştırma
```bash
g++ -O3 -std=c++17 -Wall -Wextra wrc_rally.cpp -o wrc_rally -lX11 -lGL -lGLU -lm
./wrc_rally
```

## Kontroller
- **W / S:** Gaz / Fren & Geri Vites
- **A / D:** Direksiyon
- **Space:** Hidrolik El Freni
- **C:** Kamera Açısı Değiştir
- **V:** Araç Değiştir
- **R:** Aracı Yola Sıfırla
- **K:** İklim & Ortam Değiştir
- **M:** 2D Etap Çizim Ekranına Dön
- **P:** Otopilot Demo Modu
- **ESC:** Çıkış
