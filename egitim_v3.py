"""
ML Tabanlı IDS — Eğitim Scripti v3
===================================
Marmara Üniversitesi | Veri Madenciliği Final | Haziran 2026

Kullanım:
    python egitim_v3.py

Notlar:
- CSV'ler C:/CIC-2017/ klasöründe direkt bulunuyor (alt klasör yok)
- 20 kolon (güncel liste — Min Packet Length + Active Mean dahil)
- 9 algoritma: RF, DT, XGB, KNN, SVM, NB, GB, ExtraTrees (+1), MLP (+2)
- 70/30 ve 80/20 split -> toplam 18 model
- Çıktı klasörü: C:/CIC-2017/final_modeller_v3/
"""

import os, glob, time, warnings, pickle, json, sys
import pandas as pd
import numpy as np

# Windows'ta Türkçe/özel karakter sorunu — stdout'u UTF-8'e zorla
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────
VERI_KLASORU  = r"C:\CIC-2017"
CIKTI_KLASORU = r"C:\CIC-2017\final_modeller_v5"
MAX_ORNEK     = 5000
RANDOM_STATE  = 42
ETIKET_KOLONU = "Label"

# 20 Kolon — güncellendi (Fwd Header Length -> Min Packet Length + Active Mean)
KOLONLAR = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Fwd Packet Length Max",
    "Min Packet Length",       # YENİ — SYN flood'da 0'a iner, çok ayırt edici
    "Packet Length Mean",
    "Packet Length Variance",  # En önemli kolon — %11.49 feature importance
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "Fwd PSH Flags",
    "Active Mean",             # YENİ — DDoS vs BENIGN ayrımında güçlü
]

os.makedirs(CIKTI_KLASORU, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. VERİYİ OKU
# ─────────────────────────────────────────────────────────────────────────────
log("=" * 60)
log("ADIM 1/6 — CSV dosyaları okunuyor...")
log("=" * 60)

dosyalar = glob.glob(os.path.join(VERI_KLASORU, "*.csv"))
dosyalar = [d for d in dosyalar if "temiz_veri" not in d.lower() and "test_ornekleri" not in d.lower()]  # temiz_veri.csv atla

if not dosyalar:
    raise FileNotFoundError(f"CSV bulunamadı: {VERI_KLASORU}")

parcalar = []
for dosya in dosyalar:
    log(f"  Okunuyor: {os.path.basename(dosya)}")
    okundu = False
    for enc in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
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
            log(f"  [UYARI] {os.path.basename(dosya)}: {e}")
            break
    if not okundu:
        log(f"  [ATLANDI] {os.path.basename(dosya)} hicbir encoding ile okunamadi")

df_ham = pd.concat(parcalar, ignore_index=True)
log(f"\n  Toplam ham satir: {len(df_ham):,}")
log(f"  Toplam kolon: {len(df_ham.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. KOLON SEÇİMİ VE TEMİZLİK
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 2/6 — Kolon seçimi ve temizlik...")
log("=" * 60)

# Eksik kolon kontrolü
eksik = [k for k in KOLONLAR + [ETIKET_KOLONU] if k not in df_ham.columns]
if eksik:
    log(f"\n  [UYARI] Şu kolonlar veri setinde bulunamadı:")
    for e in eksik:
        log(f"    - {e}")
    log(f"\n  Mevcut kolonlar arasında benzer isimler aranıyor...")
    for e in eksik:
        benzerler = [c for c in df_ham.columns if e.lower().split()[0] in c.lower()]
        if benzerler:
            log(f"    '{e}' için olası eşleşme: {benzerler[:3]}")
    # Eksik kolonları listeden çıkar
    KOLONLAR = [k for k in KOLONLAR if k in df_ham.columns]
    log(f"\n  Devam edilen kolon sayısı: {len(KOLONLAR)}")

df = df_ham[KOLONLAR + [ETIKET_KOLONU]].copy()

# NaN -> 0
nan_sayisi = df[KOLONLAR].isna().sum().sum()
df[KOLONLAR] = df[KOLONLAR].fillna(0)
log(f"  NaN değer temizlendi: {nan_sayisi:,}")

inf_sayisi = np.isinf(df[KOLONLAR].select_dtypes(include=[np.number]).values).sum()
df[KOLONLAR] = df[KOLONLAR].replace([np.inf, -np.inf], np.nan)
for kol in KOLONLAR:
    if df[kol].isna().any():
        cap = df[kol].quantile(0.999)
        df[kol] = df[kol].fillna(cap).clip(upper=cap * 10)
log(f"  Sonsuz deger temizlendi: {inf_sayisi:,} (percentile cap ile)")

# Etiket temizliği
df[ETIKET_KOLONU] = df[ETIKET_KOLONU].str.strip()

log(f"\n  Ham sınıf dağılımı:")
for sinif, sayi in df[ETIKET_KOLONU].value_counts().items():
    log(f"    {sinif:<35} {sayi:>10,}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DENGELİ ÖRNEKLEME
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log(f"ADIM 3/6 — Dengeli örnekleme (her sınıftan max {MAX_ORNEK})...")
log("=" * 60)

gruplar = []
for sinif, grup in df.groupby(ETIKET_KOLONU):
    n = min(len(grup), MAX_ORNEK)
    orneklem = grup.sample(n=n, random_state=RANDOM_STATE)
    gruplar.append(orneklem)
    log(f"  {sinif:<35} {len(grup):>10,}  ->  {n:>5}")

df_dengeli = pd.concat(gruplar, ignore_index=True).sample(
    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
log(f"\n  Dengeli toplam: {len(df_dengeli):,} satir")

X = df_dengeli[KOLONLAR].values
y_ham = df_dengeli[ETIKET_KOLONU].values

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 4/6 — Label encoding...")
log("=" * 60)

le = LabelEncoder()
y_enc = le.fit_transform(y_ham)
log(f"  Sınıflar ({len(le.classes_)} adet): {list(le.classes_)}")

pickle.dump(le,      open(os.path.join(CIKTI_KLASORU, "label_encoder.pkl"), "wb"))
pickle.dump(KOLONLAR, open(os.path.join(CIKTI_KLASORU, "kolonlar.pkl"),     "wb"))
log("  label_encoder.pkl ve kolonlar.pkl kaydedildi.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EĞİTİM
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 5/6 — Model eğitimi başlıyor...")
log("=" * 60)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGB_VAR = True
except ImportError:
    log("  [UYARI] xgboost yok -> pip install xgboost")
    XGB_VAR = False

# (isim, model, scaler_gerekli_mi)
algoritmalar = [
    ("RandomForest",      RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
    ("DecisionTree",      DecisionTreeClassifier(random_state=RANDOM_STATE),                              False),
    ("KNN",               KNeighborsClassifier(n_neighbors=5, n_jobs=-1),                                 True),
    ("SVM",               LinearSVC(max_iter=2000, random_state=RANDOM_STATE),                            True),
    ("NaiveBayes",        GaussianNB(),                                                                    True),
    ("GradientBoosting",  GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),        False),
    ("ExtraTrees",        ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),   False),
    ("MLP",               MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300,
                                        random_state=RANDOM_STATE),                                        True),
]
if XGB_VAR:
    algoritmalar.insert(2, ("XGBoost", XGBClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, verbosity=0), False))
SPLITLER = [("70_30", 0.30), ("80_20", 0.20)]
sonuclar = []

for split_adi, test_oran in SPLITLER:
    log(f"\n{'━'*60}")
    log(f"  SPLIT: %{'70' if test_oran==0.30 else '80'} eğitim / %{'30' if test_oran==0.30 else '20'} test")
    log(f"{'━'*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_oran, random_state=RANDOM_STATE, stratify=y_enc)
    log(f"  Eğitim seti: {len(X_train):,} | Test seti: {len(X_test):,}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(CIKTI_KLASORU, f"scaler_{split_adi}.pkl"), "wb"))

    for alg_adi, model, sc_gerekli in algoritmalar:
        log(f"\n  ▶ {alg_adi} [{split_adi}]")
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

        log(f"    Accuracy : {acc:.4f}  |  F1 : {f1:.4f}  |  Precision : {prec:.4f}  |  Recall : {rec:.4f}")
        log(f"    Eğitim   : {sure_egitim}s  |  Tahmin : {sure_tahmin}s")

        # Model kaydet
        model_yolu = os.path.join(CIKTI_KLASORU, f"model_{alg_adi}_{split_adi}.pkl")
        pickle.dump(model, open(model_yolu, "wb"))

        # Detaylı rapor
        rapor_txt = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
        rapor_yolu = os.path.join(CIKTI_KLASORU, f"rapor_{alg_adi}_{split_adi}.txt")
        with open(rapor_yolu, "w", encoding="utf-8") as f:
            f.write(f"{alg_adi} | {split_adi}\n")
            f.write(f"Accuracy:{acc}  F1:{f1}  Precision:{prec}  Recall:{rec}\n\n")
            f.write(rapor_txt)

        # Feature importance (sadece tree tabanlı)
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(KOLONLAR, model.feature_importances_.tolist()))
            fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            fi_yolu = os.path.join(CIKTI_KLASORU, f"feature_importance_{alg_adi}_{split_adi}.json")
            json.dump(fi_sorted, open(fi_yolu, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        sonuclar.append({
            "Algoritma":      alg_adi,
            "Split":          split_adi,
            "Accuracy":       acc,
            "F1_Score":       f1,
            "Precision":      prec,
            "Recall":         rec,
            "Egitim_Sure_s":  sure_egitim,
            "Tahmin_Sure_s":  sure_tahmin,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 6. SONUÇLAR
# ─────────────────────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("ADIM 6/6 — Sonuçlar kaydediliyor...")
log("=" * 60)

df_sonuc = pd.DataFrame(sonuclar).sort_values(["F1_Score"], ascending=False)
sonuc_yolu = os.path.join(CIKTI_KLASORU, "sonuclar.csv")
df_sonuc.to_csv(sonuc_yolu, index=False, encoding="utf-8-sig")

# En iyi model
en_iyi = df_sonuc.iloc[0]

log(f"\n{'═'*60}")
log("  ÖZET TABLO (F1 Skora Göre Sıralı)")
log(f"{'═'*60}")
log(f"  {'Algoritma':<20} {'Split':<8} {'Accuracy':>10} {'F1':>8} {'Precision':>11} {'Recall':>8}")
log(f"  {'-'*65}")
for _, row in df_sonuc.iterrows():
    log(f"  {row['Algoritma']:<20} {row['Split']:<8} {row['Accuracy']:>10.4f} "
        f"{row['F1_Score']:>8.4f} {row['Precision']:>11.4f} {row['Recall']:>8.4f}")

log(f"\n{'═'*60}")
log(f"  EN İYİ MODEL : {en_iyi['Algoritma']} [{en_iyi['Split']}]")
log(f"  F1 Score     : {en_iyi['F1_Score']:.4f}")
log(f"  Accuracy     : {en_iyi['Accuracy']:.4f}")
log(f"{'═'*60}")
log(f"\n✓ Tüm dosyalar: {CIKTI_KLASORU}")
log(f"✓ sonuclar.csv hazır -> rapora yapıştır")
log(f"✓ En iyi modeli Ubuntu VM'e gönder: model_{en_iyi['Algoritma']}_{en_iyi['Split']}.pkl")