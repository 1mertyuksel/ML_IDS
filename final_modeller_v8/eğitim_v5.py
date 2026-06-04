"""
ML Tabanli IDS - Egitim Scripti v5
====================================
Marmara Universitesi | Veri Madenciligi Final | Haziran 2026

Degisiklikler v4'e gore:
- Kendi veri seti kullaniliyor (kendi_dataset.csv)
- 33 kolon — Random Forest Feature Importance ile secildi (importance > 0.01)
- MAX_ORNEK = 4000 (dengeli ornekleme)
- 6 sinif: BENIGN, DoS-DDoS, PortScan, SSH-Patator, FTP-Patator, Web-BruteForce
- Cikti: C:/CIC-2017/final_modeller_v8/

Kolon Secim Yontemi:
- 84 ham ozellik arasından RF Feature Importance hesaplandi
- Gini impurity azaltma katki degeri > 0.01 olan 33 ozellik secildi
- Bu esik altindaki ozellikler overfitting riski tasidiginden cikarildi
"""

import os, glob, time, warnings, pickle, json, sys
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
VERI_DOSYASI  = r"C:\CIC-2017\kendi_dataset.csv"
CIKTI_KLASORU = r"C:\CIC-2017\final_modeller_v8"
MAX_ORNEK     = 4000
RANDOM_STATE  = 42
ETIKET_KOLONU = "Label"

# 33 KOLON — Random Forest Feature Importance > 0.01
# Her kolonun secilme gerekce si asagida aciklanmistir
KOLONLAR = [
    "Dst Port",                  # Hedef port — PortScan, FTP, SSH siniflarini ayirt eder
    "Fwd Seg Size Min",          # Min segment boyutu — saldiri tipine gore degisir
    "Packet Length Max",         # Max paket boyutu — DoS ve BENIGN'i ayirt eder
    "Bwd Packet Length Mean",    # Geri yon ort paket boyutu — sunucu yanit tipi
    "Bwd Packet Length Std",     # Geri yon std — trafik duzenliligini gosterir
    "ACK Flag Count",            # ACK sayisi — FTP'de cok yuksek (18), PortScan'da dusuk
    "Total Length of Bwd Packet",# Toplam geri trafik hacmi
    "Bwd Packet Length Max",     # Max geri paket — BENIGN'de buyuk HTTP yaniti
    "Packet Length Std",         # Paket boyutu tutarsizligi
    "Packet Length Mean",        # Ortalama paket boyutu
    "Subflow Fwd Bytes",         # Alt akis ileri byte — trafik yogunlugu
    "Bwd Segment Size Avg",      # Ortalama geri segment boyutu
    "Subflow Bwd Bytes",         # Alt akis geri byte
    "Packet Length Variance",    # Paket boyutu varyasyonu — trafik cesitliligi
    "FIN Flag Count",            # Baglanti kapatma — normal trafik vs saldiri
    "Total Length of Fwd Packet",# Toplam ileri trafik hacmi
    "Flow IAT Std",              # Paketler arasi sure std — duzensizlik olcusu
    "Bwd Header Length",         # Geri baslik uzunlugu
    "Fwd Packet Length Max",     # Max ileri paket boyutu
    "Subflow Fwd Packets",       # Alt akis ileri paket sayisi
    "Bwd Packets/s",             # Geri yon paket hizi
    "Subflow Bwd Packets",       # Alt akis geri paket sayisi
    "SYN Flag Count",            # Baglanti baslama — PortScan ve brute force
    "Flow Duration",             # Akis suresi — en guclu ayirt edici
    "Total Fwd Packet",          # Toplam ileri paket — trafik yogunlugu
    "RST Flag Count",            # Baglanti reddi — PortScan ve SSH'ta gorulur
    "Fwd Header Length",         # Ileri baslik uzunlugu
    "Flow Packets/s",            # Toplam paket hizi — PortScan=24096, DoS=4
    "Fwd Packet Length Std",     # Ileri paket boyutu std
    "FWD Init Win Bytes",        # Baslangic pencere boyutu — OS parmak izi
    "Flow Bytes/s",              # Byte hizi — volumetrik saldirilar
    "Bwd Bulk Rate Avg",         # Geri bulk transfer hizi
    "Flow IAT Mean",             # Paketler arasi ortalama sure — saldiri ritmi
]

os.makedirs(CIKTI_KLASORU, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. VERI OKU
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("ADIM 1/6 - Kendi veri seti okunuyor...")
log("=" * 60)

if not os.path.exists(VERI_DOSYASI):
    raise FileNotFoundError(f"Veri seti bulunamadi: {VERI_DOSYASI}")

df_ham = pd.read_csv(VERI_DOSYASI, low_memory=False)
df_ham.columns = df_ham.columns.str.strip()
log(f"  Toplam satir: {len(df_ham):,}")
log(f"  Toplam kolon: {len(df_ham.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. KOLON SECIMI VE TEMIZLIK
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 2/6 - Kolon secimi ve temizlik...")
log("=" * 60)

eksik = [k for k in KOLONLAR if k not in df_ham.columns]
if eksik:
    log(f"  [UYARI] Eksik kolonlar: {eksik}")
    KOLONLAR_KULLAN = [k for k in KOLONLAR if k in df_ham.columns]
else:
    KOLONLAR_KULLAN = KOLONLAR

df = df_ham[KOLONLAR_KULLAN + [ETIKET_KOLONU]].copy()

nan_sayisi = df[KOLONLAR_KULLAN].isna().sum().sum()
df[KOLONLAR_KULLAN] = df[KOLONLAR_KULLAN].fillna(0)
log(f"  NaN temizlendi: {nan_sayisi:,}")

inf_toplam = 0
for kol in KOLONLAR_KULLAN:
    inf_mask = np.isinf(df[kol].values)
    inf_sayisi_kol = inf_mask.sum()
    if inf_sayisi_kol > 0:
        gecerli = df.loc[~inf_mask, kol]
        if len(gecerli) > 0:
            cap = gecerli.quantile(0.995)
            df.loc[inf_mask, kol] = cap
        inf_toplam += inf_sayisi_kol

log(f"  inf temizlendi: {inf_toplam:,}")
log(f"\n  Ham sinif dagilimi:")
for sinif, sayi in df[ETIKET_KOLONU].value_counts().items():
    log(f"    {sinif:<35} {sayi:>10,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DENGELI ORNEKLEME
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log(f"ADIM 3/6 - Dengeli ornekleme (her siniftan max {MAX_ORNEK})...")
log("=" * 60)

gruplar = []
for sinif, grup in df.groupby(ETIKET_KOLONU):
    n = min(len(grup), MAX_ORNEK)
    orneklem = grup.sample(n=n, random_state=RANDOM_STATE)
    gruplar.append(orneklem)
    log(f"  {sinif:<35} {len(grup):>10,} -> {n:>5}")

df_dengeli = pd.concat(gruplar, ignore_index=True).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
log(f"\n  Dengeli toplam: {len(df_dengeli):,} satir")

X = df_dengeli[KOLONLAR_KULLAN].values
y_ham = df_dengeli[ETIKET_KOLONU].values

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 4/6 - Label encoding...")
log("=" * 60)

le = LabelEncoder()
y_enc = le.fit_transform(y_ham)
log(f"  Siniflar ({len(le.classes_)} adet): {list(le.classes_)}")

pickle.dump(le,             open(os.path.join(CIKTI_KLASORU, "label_encoder.pkl"), "wb"))
pickle.dump(KOLONLAR_KULLAN,open(os.path.join(CIKTI_KLASORU, "kolonlar.pkl"),     "wb"))

kolon_stats = {}
for kol in KOLONLAR_KULLAN:
    kolon_stats[kol] = {
        "mean": float(df_dengeli[kol].mean()),
        "std":  float(df_dengeli[kol].std()),
        "p995": float(df_dengeli[kol].quantile(0.995)),
        "max":  float(df_dengeli[kol].max()),
    }
json.dump(kolon_stats, open(os.path.join(CIKTI_KLASORU, "kolon_stats.json"), "w"),
          ensure_ascii=False, indent=2)
log("  label_encoder.pkl, kolonlar.pkl, kolon_stats.json kaydedildi.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ALGORITMALAR VE EGITIM
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 5/6 - Model egitimi basliyor...")
log("=" * 60)

try:
    from xgboost import XGBClassifier
    XGB_VAR = True
except ImportError:
    log("  [UYARI] xgboost yok")
    XGB_VAR = False

algoritmalar = [
    ("RandomForest",     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
    ("DecisionTree",     DecisionTreeClassifier(random_state=RANDOM_STATE),                              False),
    ("KNN",              KNeighborsClassifier(n_neighbors=5, n_jobs=-1),                                 True),
    ("SVM",              LinearSVC(max_iter=2000, random_state=RANDOM_STATE),                            True),
    ("NaiveBayes",       GaussianNB(),                                                                    True),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),        False),
    ("ExtraTrees",       ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),   False),
    ("MLP",              MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300,
                                       random_state=RANDOM_STATE),                                        True),
]
if XGB_VAR:
    algoritmalar.insert(2, ("XGBoost", XGBClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss",
        n_jobs=-1, verbosity=0), False))

SPLITLER = [("70_30", 0.30), ("80_20", 0.20)]
sonuclar = []

for split_adi, test_oran in SPLITLER:
    log(f"\n{'='*60}")
    log(f"  SPLIT: %{'70' if test_oran==0.30 else '80'} egitim / %{'30' if test_oran==0.30 else '20'} test")
    log(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_oran, random_state=RANDOM_STATE, stratify=y_enc)
    log(f"  Egitim: {len(X_train):,} | Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(CIKTI_KLASORU, f"scaler_{split_adi}.pkl"), "wb"))

    for alg_adi, model, sc_gerekli in algoritmalar:
        log(f"\n  [{alg_adi}] [{split_adi}]")
        Xtr = X_tr_sc if sc_gerekli else X_train
        Xte = X_te_sc if sc_gerekli else X_test

        t0 = time.time()
        model.fit(Xtr, y_train)
        sure_egitim = round(time.time() - t0, 2)

        t1 = time.time()
        y_pred = model.predict(Xte)
        sure_tahmin = round(time.time() - t1, 4)

        acc  = round(accuracy_score(y_test, y_pred), 4)
        f1   = round(f1_score(y_test, y_pred, average="weighted"), 4)
        prec = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        rec  = round(recall_score(y_test, y_pred, average="weighted"), 4)

        log(f"    Accuracy:{acc:.4f}  F1:{f1:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}")
        log(f"    Egitim:{sure_egitim}s  Tahmin:{sure_tahmin}s")

        model_yolu = os.path.join(CIKTI_KLASORU, f"model_{alg_adi}_{split_adi}.pkl")
        pickle.dump(model, open(model_yolu, "wb"))

        rapor = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
        rapor_yolu = os.path.join(CIKTI_KLASORU, f"rapor_{alg_adi}_{split_adi}.txt")
        with open(rapor_yolu, "w", encoding="utf-8") as f:
            f.write(f"{alg_adi} | {split_adi}\n")
            f.write(f"Accuracy:{acc}  F1:{f1}  Prec:{prec}  Rec:{rec}\n\n")
            f.write(rapor)

        if hasattr(model, "feature_importances_"):
            fi = dict(zip(KOLONLAR_KULLAN, model.feature_importances_.tolist()))
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            json.dump(fi_sorted, open(os.path.join(CIKTI_KLASORU,
                      f"fi_{alg_adi}_{split_adi}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

        sonuclar.append({
            "Algoritma": alg_adi, "Split": split_adi,
            "Accuracy": acc, "F1_Score": f1,
            "Precision": prec, "Recall": rec,
            "Egitim_s": sure_egitim, "Tahmin_s": sure_tahmin,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 6. SONUCLAR
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 6/6 - Sonuclar kaydediliyor...")
log("=" * 60)

df_sonuc = pd.DataFrame(sonuclar).sort_values("F1_Score", ascending=False)
df_sonuc.to_csv(os.path.join(CIKTI_KLASORU, "sonuclar.csv"), index=False, encoding="utf-8-sig")

log(f"\n{'='*60}")
log("  OZET TABLO (F1 Skora Gore Sirali)")
log(f"{'='*60}")
log(f"  {'Algoritma':<20} {'Split':<8} {'Accuracy':>10} {'F1':>8} {'Prec':>8} {'Rec':>8}")
log(f"  {'-'*60}")
for _, row in df_sonuc.iterrows():
    log(f"  {row['Algoritma']:<20} {row['Split']:<8} {row['Accuracy']:>10.4f} "
        f"{row['F1_Score']:>8.4f} {row['Precision']:>8.4f} {row['Recall']:>8.4f}")

en_iyi = df_sonuc.iloc[0]
log(f"\n{'='*60}")
log(f"  EN IYI MODEL : {en_iyi['Algoritma']} [{en_iyi['Split']}]")
log(f"  F1 Score     : {en_iyi['F1_Score']:.4f}")
log(f"  Accuracy     : {en_iyi['Accuracy']:.4f}")
log(f"{'='*60}")
log(f"\n  Tum dosyalar: {CIKTI_KLASORU}")
log(f"  Canli test icin: model_{en_iyi['Algoritma']}_{en_iyi['Split']}.pkl")