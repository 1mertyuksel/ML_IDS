"""
ML Tabanli IDS - Egitim Scripti v4
====================================
Marmara Universitesi | Veri Madenciligi Final | Haziran 2026

Degisiklikler v3'e gore:
- 13 TEMIZ kolon (NaN < %1 olan kolonlar)
- inf duzeltme: duration=0 olan flow'larda min 1000us kullanilir
- Flow Bytes/s ve Flow Packets/s icin per-column percentile cap
- Cikti: C:/CIC-2017/final_modeller_v6/

Kullanim:
    python egitim_v4.py
"""

import os, glob, time, warnings, pickle, json, sys
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
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

# Windows CP1254 sorunu
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
VERI_KLASORU  = r"C:\CIC-2017"
CIKTI_KLASORU = r"C:\CIC-2017\final_modeller_v6"
MAX_ORNEK     = 5000
RANDOM_STATE  = 42
ETIKET_KOLONU = "Label"

# 13 TEMIZ KOLON — NaN orani <%1, tum dosyalarda mevcut
KOLONLAR = [
    "Flow Duration",           # us — siniflar arasi en guclu ayirti
    "Fwd Packet Length Mean",  # ileri yon ort paket boyutu
    "Bwd Packet Length Mean",  # geri yon ort paket boyutu — DDoS'ta cok yuksek
    "Flow Bytes/s",            # volumetrik saldirilarda yuksek
    "Flow Packets/s",          # flood saldirilarda cok yuksek
    "Flow IAT Mean",           # paketler arasi sure — PortScan'da dusuk
    "Flow IAT Std",            # duzensizlik olcusu
    "Fwd PSH Flags",           # HTTP/uygulama katmani
    "SYN Flag Count",          # SYN flood icin
    "RST Flag Count",          # PortScan'da RST cevaplari
    "ACK Flag Count",          # oturum tipi
    "Avg Fwd Segment Size",    # Fwd Packet Length Mean ile ayni deger, dogrulama
    "Avg Bwd Segment Size",    # Bwd Packet Length Mean ile ayni deger, dogrulama
]

os.makedirs(CIKTI_KLASORU, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. VERI OKU
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("ADIM 1/6 - CSV dosyalari okunuyor...")
log("=" * 60)

dosyalar = glob.glob(os.path.join(VERI_KLASORU, "*.csv"))
dosyalar = [d for d in dosyalar
            if "temiz_veri" not in d.lower()
            and "test_ornekleri" not in d.lower()
            and "final_modeller" not in d.lower()]

if not dosyalar:
    raise FileNotFoundError(f"CSV bulunamadi: {VERI_KLASORU}")

parcalar = []
for dosya in dosyalar:
    log(f"  Okunuyor: {os.path.basename(dosya)}")
    okundu = False
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df_parca = pd.read_csv(dosya, encoding=enc, low_memory=False)
            df_parca.columns = df_parca.columns.str.strip()
            parcalar.append(df_parca)
            log(f"    -> {len(df_parca):,} satir [{enc}]")
            okundu = True
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log(f"  [UYARI] {e}")
            break
    if not okundu:
        log(f"  [ATLANDI] {os.path.basename(dosya)}")

df_ham = pd.concat(parcalar, ignore_index=True)
log(f"\n  Toplam ham satir: {len(df_ham):,}")
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
    KOLONLAR = [k for k in KOLONLAR if k in df_ham.columns]

df = df_ham[KOLONLAR + [ETIKET_KOLONU]].copy()

# NaN temizle
nan_sayisi = df[KOLONLAR].isna().sum().sum()
df[KOLONLAR] = df[KOLONLAR].fillna(0)
log(f"  NaN temizlendi: {nan_sayisi:,}")

# inf temizle — per-column percentile cap
# Flow Bytes/s ve Flow Packets/s'de cok fazla inf var
# Neden: Flow Duration = 0 olan satir -> bolme sonsuz
# Cozum: her kolon icin 99.5. percentile degerini cap olarak kullan
inf_toplam = 0
for kol in KOLONLAR:
    inf_mask = np.isinf(df[kol].values)
    inf_sayisi_kol = inf_mask.sum()
    if inf_sayisi_kol > 0:
        # Inf olmayan degerlerin 99.5. percentile'ini bul
        gecerli = df.loc[~inf_mask, kol]
        if len(gecerli) > 0:
            cap = gecerli.quantile(0.995)
            df.loc[inf_mask, kol] = cap
            log(f"  inf->cap: {kol}: {inf_sayisi_kol:,} deger -> {cap:.2f}")
        inf_toplam += inf_sayisi_kol

log(f"  Toplam inf temizlendi: {inf_toplam:,}")

# Etiket temizligi
df[ETIKET_KOLONU] = df[ETIKET_KOLONU].str.strip()
# Web Attack encoding duzeltmesi
df[ETIKET_KOLONU] = df[ETIKET_KOLONU].str.replace('ï¿½', '-', regex=False)
df[ETIKET_KOLONU] = df[ETIKET_KOLONU].str.replace('�', '-', regex=False)

log(f"\n  Ham sinif dagilimi:")
for sinif, sayi in df[ETIKET_KOLONU].value_counts().items():
    log(f"    {sinif:<40} {sayi:>10,}")

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
    log(f"  {sinif:<40} {len(grup):>10,} -> {n:>5}")

df_dengeli = pd.concat(gruplar, ignore_index=True).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
log(f"\n  Dengeli toplam: {len(df_dengeli):,} satir")

X = df_dengeli[KOLONLAR].values
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

pickle.dump(le,      open(os.path.join(CIKTI_KLASORU, "label_encoder.pkl"), "wb"))
pickle.dump(KOLONLAR, open(os.path.join(CIKTI_KLASORU, "kolonlar.pkl"),     "wb"))
log("  label_encoder.pkl ve kolonlar.pkl kaydedildi.")

# Kolon istatistiklerini kaydet (canlida ayni cap degerlerini kullanmak icin)
kolon_stats = {}
for kol in KOLONLAR:
    kolon_stats[kol] = {
        "mean": float(df_dengeli[kol].mean()),
        "std":  float(df_dengeli[kol].std()),
        "p995": float(df_dengeli[kol].quantile(0.995)),
        "max":  float(df_dengeli[kol].max()),
    }
json.dump(kolon_stats, open(os.path.join(CIKTI_KLASORU, "kolon_stats.json"), "w"),
          ensure_ascii=False, indent=2)
log("  kolon_stats.json kaydedildi.")

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
    log("  [UYARI] xgboost yok -> pip install xgboost")
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
            fi = dict(zip(KOLONLAR, model.feature_importances_.tolist()))
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            fi_yolu = os.path.join(CIKTI_KLASORU, f"fi_{alg_adi}_{split_adi}.json")
            json.dump(fi_sorted, open(fi_yolu, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

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
sonuc_yolu = os.path.join(CIKTI_KLASORU, "sonuclar.csv")
df_sonuc.to_csv(sonuc_yolu, index=False, encoding="utf-8-sig")

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
log(f"  sonuclar.csv hazir")
log(f"  Canli test icin: model_{en_iyi['Algoritma']}_{en_iyi['Split']}.pkl")