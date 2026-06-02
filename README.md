# 🛡️ ML Tabanlı Saldırı Tespit Sistemi (IDS)

**Marmara Üniversitesi — Bilgisayar Mühendisliği**  
**Veri Madenciliği Final Projesi | Haziran 2026**  
**Danışman:** Doç. Dr. Ayşe Berna Altınel Girgin

---

## 👥 Grup Üyeleri

| İsim | Rol |
|---|---|
| Çağatay Sofu | |
| Oğuzhan Bozkurt | |
| Mehmet Fatih İnan | |
| Fatih Demirbaş | |
| Mert | |

---

## 📌 Proje Özeti

Bu çalışmada **CIC-IDS-2017** veri seti kullanılarak makine öğrenmesi tabanlı bir **Saldırı Tespit Sistemi (IDS)** geliştirilmiştir.

- **20 ağ trafiği özniteliği** üzerinde 9 farklı algoritma karşılaştırmalı olarak eğitilmiştir
- **%70/%30 ve %80/%20** eğitim/test bölme oranları denenmiştir
- En iyi model, **Kali Linux → Ubuntu Server VM** ortamında canlı saldırı testi ile doğrulanmıştır

---

## 📊 Deney Sonuçları

| Algoritma | F1 [70/30] | F1 [80/20] | Accuracy [80/20] |
|---|---|---|---|
| **DecisionTree** | 0.9784 | **0.9792** | 0.9789 |
| **RandomForest** | 0.9785 | **0.9789** | 0.9789 |
| **XGBoost** | 0.9780 | **0.9789** | 0.9816 |
| ExtraTrees (+1) | 0.9779 | 0.9781 | 0.9780 |
| GradientBoosting | 0.9773 | 0.9768 | 0.9788 |
| KNN | 0.9678 | 0.9678 | 0.9686 |
| MLP (+2) | 0.9609 | 0.9603 | 0.9650 |
| SVM | 0.8700 | 0.8689 | 0.8787 |
| NaiveBayes | 0.6632 | 0.6634 | 0.6767 |

> **En iyi model:** DecisionTree [80/20] F1=0.9792  
> **Canlı test modeli:** XGBoost [80/20] (overfitting riski olmadan kararlı performans)

---

## 📁 Dosya Yapısı

```
CIC-IDS-2017/
│
├── egitim_v3.py                  # Ana eğitim scripti
│
├── final_modeller_v5/
│   ├── kolonlar.pkl              # 20 kolon listesi (zorunlu)
│   ├── label_encoder.pkl         # Sınıf ismi ↔ sayı eşleşmesi (zorunlu)
│   ├── scaler_70_30.pkl          # StandardScaler — 70/30 split
│   ├── scaler_80_20.pkl          # StandardScaler — 80/20 split (canlı test)
│   ├── sonuclar.csv              # Tüm model sonuçları
│   │
│   ├── model_XGBoost_80_20.pkl       ✅ Canlı test modeli
│   ├── model_DecisionTree_80_20.pkl  ✅
│   ├── model_GradientBoosting_*.pkl  ✅
│   ├── model_MLP_*.pkl               ✅
│   ├── model_SVM_*.pkl               ✅
│   ├── model_NaiveBayes_*.pkl        ✅
│   │
│   ├── model_RandomForest_*.pkl      ⚠️  Büyük (32MB) — aşağıya bakın
│   ├── model_ExtraTrees_*.pkl        ❌  Çok büyük (80MB) — aşağıya bakın
│   ├── model_KNN_*.pkl               ⚠️  Orta (6MB)
│   │
│   ├── rapor_*.txt               # Her model için sınıf bazlı rapor (18 adet)
│   └── feature_importance_*.json # Kolon önem skorları (ağaç modelleri için)
│
└── README.md
```

> ⚠️ **RandomForest ve ExtraTrees modelleri GitHub'a yüklenmemiştir** (boyut sınırı).  
> Bu modelleri kendiniz üretmek için aşağıdaki kurulum adımlarını izleyin.

---

## ⚙️ Kurulum

### 1. Gereksinimler

```bash
pip install scikit-learn xgboost pandas numpy
```

### 2. Veri Seti

[CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) veri setini indirin.  
CSV dosyalarını `C:\CIC-2017\` klasörüne koyun.

Beklenen dosyalar:
```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

### 3. Eğitimi Çalıştır

```bash
cd C:\CIC-2017
python -u egitim_v3.py
```

Çıktılar `C:\CIC-2017\final_modeller_v5\` klasörüne kaydedilir.  
Tahmini süre: ~10 dakika (GradientBoosting dahil)

---

## 🔬 Yöntem

### Veri Hazırlama

- **Dengeli örnekleme:** Her sınıftan maksimum 5.000 örnek (toplam ~49.193)
- **NaN temizleme:** → 0
- **Sonsuz değer temizleme:** → kolon bazlı 99.9. yüzdelik dilim
- **Label Encoding:** Sınıf isimleri sayıya çevrildi
- **StandardScaler:** Yalnızca KNN, SVM, NaiveBayes, MLP için

### Kullanılan 20 Öznitelik

| # | Öznitelik | Açıklama |
|---|---|---|
| 1 | Destination Port | Hedef port |
| 2 | Flow Duration | Akış süresi |
| 3 | Total Fwd Packets | İleri yön paket sayısı |
| 4 | Total Backward Packets | Geri yön paket sayısı |
| 5 | Fwd Packet Length Mean | İleri yön ort. paket boyutu |
| 6 | Bwd Packet Length Mean | Geri yön ort. paket boyutu |
| 7 | Fwd Packet Length Max | İleri yön maks. paket boyutu |
| 8 | Min Packet Length | Minimum paket boyutu |
| 9 | Packet Length Mean | Genel ort. paket boyutu |
| 10 | **Packet Length Variance** | **Paket boyutu varyansı (en önemli öznitelik)** |
| 11 | Flow Bytes/s | Saniyedeki byte |
| 12 | Flow Packets/s | Saniyedeki paket |
| 13 | Flow IAT Mean | Paketler arası ort. süre |
| 14 | Flow IAT Std | Paketler arası süre std. sapması |
| 15 | Fwd IAT Mean | İleri yön paketler arası ort. |
| 16 | SYN Flag Count | SYN bayrağı sayısı |
| 17 | RST Flag Count | RST bayrağı sayısı |
| 18 | ACK Flag Count | ACK bayrağı sayısı |
| 19 | Fwd PSH Flags | PSH bayrağı sayısı |
| 20 | Active Mean | Akışın aktif süresi ortalaması |

### Algoritmalar

| # | Algoritma | Tür | Not |
|---|---|---|---|
| 1 | Random Forest | Ensemble | Ana model |
| 2 | Decision Tree | Ağaç | Hızlı, yorumlanabilir |
| 3 | XGBoost | Boosting | En iyi accuracy |
| 4 | KNN | Instance-based | Scaler zorunlu |
| 5 | SVM (LinearSVC) | Kernel | Scaler zorunlu |
| 6 | Naive Bayes | Probabilistik | Scaler zorunlu |
| 7 | Gradient Boosting | Boosting | Yavaş eğitim |
| +1 | Extra Trees | Ensemble | Ek algoritma |
| +2 | MLP | Yapay Sinir Ağı | Ek algoritma, Scaler zorunlu |

---

## 🖥️ Canlı Test Ortamı

```
Kali Linux VM          Ubuntu Server VM
192.168.56.101    →    192.168.56.102
(Saldırgan)            (Kurban + IDS)
                        Apache + tcpdump
                        CICFlowMeter
                        Python + Model
```

### Test Senaryoları

| Test | Komut | Beklenen Sonuç |
|---|---|---|
| Normal trafik | `curl http://192.168.56.102/` | BENIGN |
| Port tarama | `nmap -sS -p 1-1000 192.168.56.102` | PortScan |
| SYN Flood | `sudo hping3 -S --flood -p 80 192.168.56.102` | DDoS |
| HTTP Flood | `ab -n 50000 -c 500 http://192.168.56.102/` | DDoS |
| SSH BruteForce | `hydra -L users.txt -P pass.txt ssh://192.168.56.102` | SSH-Patator |

---

## 📦 Tespit Edilen Sınıflar (15 adet)

`BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`,  
`DoS slowloris`, `FTP-Patator`, `Heartbleed`, `Infiltration`, `PortScan`,  
`SSH-Patator`, `Web Attack – Brute Force`, `Web Attack – Sql Injection`, `Web Attack – XSS`

---

## 📄 Lisans

Bu proje akademik amaçlıdır. Marmara Üniversitesi Veri Madenciliği dersi kapsamında hazırlanmıştır.
