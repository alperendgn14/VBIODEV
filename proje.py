import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Dosyayı okuyoruz (Dizini ayarlamıştın zaten)
df = pd.read_excel('veri.xlsx')

print("Sütunlar için veri temizliği.")
ilk_satir = len(df)

# 1. null'ları atma
df = df.dropna()

# Float64 için virgülü nokta ile değiştiriyorum.
df['fare_per_mile'] = df['fare_per_mile'].astype(str).str.replace(',', '.')
df['fare_per_mile'] = pd.to_numeric(df['fare_per_mile'], errors='coerce')
df = df.dropna()



# Metinli kategorilerde boşluklu kısımların temizliği
# Veri setindeki şehirler ve havayolu kodlarında veri girilirken yanlışlıkla 
# başa/sona konmuş boşluklar olabilir (' AA ' ile 'AA' farklı algılanabilir).
# Boşlukları siliyorum ve hepsini büyük harf yapıyorum.
metin_sutunlari = ['origin_city', 'destination_city', 'largest_carrier', 'lowest_fare_carrier', 'year_quarter']

for sutun in metin_sutunlari:
    df[sutun] = df[sutun].astype(str).str.strip().str.upper()


# Filtreleme
# Bilet fiyatları, mesafe ve yolcu sayısı 0 veya eksi (-) olamaz.
df = df[df['avg_fare'] > 0]
df = df[df['largest_carrier_fare'] > 0]
df = df[df['lowest_fare'] > 0]
df = df[df['distance_miles'] > 0]
df = df[df['passengers'] > 0]

# Pazar payı oranları 0.xx formatında verilmiş. 
# Bu değerler 0 ve 1 arasında olmalı.
df = df[(df['largest_carrier_market_share'] >= 0) & (df['largest_carrier_market_share'] <= 1)]
df = df[(df['lowest_fare_carrier_share'] >= 0) & (df['lowest_fare_carrier_share'] <= 1)]


# Aykırı değerlerin temizliği
def aykiri_deger_sil(dataframe, sutun):
    Q1 = dataframe[sutun].quantile(0.25)
    Q3 = dataframe[sutun].quantile(0.75)
    IQR = Q3 - Q1
    # IQR yöntemine göre alt ve üst sınırları belirliyoruz
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    return dataframe[(dataframe[sutun] >= alt_sinir) & (dataframe[sutun] <= ust_sinir)]

# Tüm sayısal sütunlar için aykırı değer temizliği
sayisal_sutunlar_icin = ['avg_fare', 'distance_miles', 'passengers', 'largest_carrier_fare', 'lowest_fare']
for sutun in sayisal_sutunlar_icin:
    df = aykiri_deger_sil(df, sutun)



# --- TEMİZLİK SONUCU ---
son_satir = len(df)
print(f"Başlangıçtaki ham satır sayısı: {ilk_satir}")
print(f"Temizlenen (hatalı, mantıksız, aykırı veya boş) satır sayısı: {ilk_satir - son_satir}")
print(f"Filtreleme sonrası kalan satır sayısı: {son_satir}")




warnings.filterwarnings('ignore') # Konsoldaki gereksiz uyarıları gizler

print("\n" + "="*50)
print("Görselleştirme")
print("="*50)

# ısı haritası ile sayısal veriler arasındaki ilişkiler
plt.figure(figsize=(8, 6))
sayisal_df = df[['distance_miles', 'passengers', 'avg_fare', 'largest_carrier_fare', 'lowest_fare']]
korelasyon_matrisi = sayisal_df.corr()
sns.heatmap(korelasyon_matrisi, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Sayısal Değişkenler Arası Korelasyon Isı Haritası')
plt.show()

# en yoğun 5 havayolunun fiyat dağılımı.
plt.figure(figsize=(10, 6))
en_buyukler = df['largest_carrier'].value_counts().nlargest(5).index
sns.boxplot(x='largest_carrier', y='avg_fare', data=df[df['largest_carrier'].isin(en_buyukler)], palette='Set2')
plt.title('En fazla uçuş yapan 5 havayolunun ortalama bilet fiyatı dağılımı')
plt.xlabel('Havayolu şirketi')
plt.ylabel('Dolar cinsinden ortalama bilet fiyatı')
plt.show()

# uçuş mesafesi / fiyat ilişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_miles', y='avg_fare', data=df, alpha=0.4, color='purple')
plt.title('Uçuş mesafesi ve Ortalama bilet fiyatı ilişkisi')
plt.xlabel('Mil cinsinden uçuş mesafesi')
plt.ylabel('Dolar cinsinden ortalama bilet fiyatı')
plt.show()


print("\n" + "="*50)
print("Hipotez testleri")
print("="*50)

# 1. Hipotez - korelasyon testi.
# H0: Mesafe ile ortalama fiyat arasında anlamlı bir ilişki yoktur.
# H1: Mesafe ile ortalama fiyat arasında anlamlı bir ilişki vardır.
print("\n--- Hipotez 1: Mesafe ve Fiyat İlişkisi ---")
r_stat, p_val_1 = stats.pearsonr(df['distance_miles'], df['avg_fare'])
print(f"Test İstatistiği (r): {r_stat:.3f}, P-Değeri: {p_val_1:.5f}")
if p_val_1 < 0.05:
    print("Sonuç (H0 Reddedilir): Uçuş mesafesi ile ortalama bilet fiyatı arasında istatistiksel olarak anlamlı&pozitif bir ilişki vardır.")
else:
    print("Sonuç (H0 Reddedilemez): Anlamlı bir ilişki bulunamamıştır.")


# 2. Hipotez - bağımsız çift örneklem t-testi
# H0: WN ve DL havayollarının ortalama fiyatları arasında fark yoktur.
# H1: İki havayolunun fiyatları arasında fark vardır.
print("\n--- Hipotez 2: WN ve DL Havayolları Fiyat Karşılaştırması (T-Test) ---")
wn_fiyat = df[df['largest_carrier'] == 'WN']['avg_fare']
dl_fiyat = df[df['largest_carrier'] == 'DL']['avg_fare']
t_stat, p_val_2 = stats.ttest_ind(wn_fiyat, dl_fiyat, equal_var=False)
print(f"Test İstatistiği (t): {t_stat:.3f}, P-Değeri: {p_val_2:.5f}")
if p_val_2 < 0.05:
    print("Sonuç (H0 Reddedilir): WN ve DL havayollarının bilet fiyatları ortalamaları arasında istatistiksel olarak anlamlı fark vardır.")
else:
    print("Sonuç (H0 Reddedilemez): Fiyatlar arasında anlamlı bir fark yoktur.")


# 3. hipotez - tek yönlü varyans analizi (anova)
# h0: en büyük 3 havayolunun (WN, DL ve AA) taşıdığı ortalama yolcu sayıları eşittir.
# h1: en az birinin taşıdığı ortalama yolcu sayısı diğerlerinden farklıdır.
print("\n--- Hipotez 3: WN, DL ve AA Yolcu Sayısı Karşılaştırması (ANOVA) ---")
wn_yolcu = df[df['largest_carrier'] == 'WN']['passengers']
dl_yolcu = df[df['largest_carrier'] == 'DL']['passengers']
aa_yolcu = df[df['largest_carrier'] == 'AA']['passengers']
f_stat, p_val_3 = stats.f_oneway(wn_yolcu, dl_yolcu, aa_yolcu)
print(f"Test İstatistiği (F): {f_stat:.3f}, P-Değeri: {p_val_3:.5f}")
if p_val_3 < 0.05:
    print("Sonuç (H0 Reddedilir): Bu üç havayolunun taşıdığı ortalama yolcu sayıları birbirinden anlamlı şekilde farklıdır")
else:
    print("Sonuç (H0 Reddedilemez): Yolcu sayıları arasında anlamlı bir fark yoktur.")


# =====================================================================
# dağılım görselleştirme / normallik analizi
# =====================================================================

print("\n" + "="*50)
print("normallik analizi")
print("="*50)

# 1. GÖRSEL EKLEMESİ: Ortalama Bilet Fiyatı Dağılım Grafiği (Histogram + KDE Eğrisi)
plt.figure(figsize=(10, 6))
sns.histplot(df['avg_fare'], bins=40, kde=True, color='teal', edgecolor='black')
plt.title('Ortalama Bilet Fiyatlarının Normal Dağılım İncelemesi')
plt.xlabel('Ortalama Bilet Fiyatı')
plt.ylabel('Frekans (Kayıt Sayısı)')
# Ortanca ve Ortalama çizgilerini ekleyelim (Dağılımın çarpıklığını gösterir)
plt.axvline(df['avg_fare'].mean(), color='red', linestyle='--', label=f"Ortalama: {df['avg_fare'].mean():.1f}$")
plt.axvline(df['avg_fare'].median(), color='orange', linestyle='-', label=f"Medyan: {df['avg_fare'].median():.1f}$")
plt.legend()
plt.show()

# shapiro normallik testi
# H0: Bilet fiyatları normal dağılıma uymaktadır.
# H1: Bilet fiyatları normal dağılıma uymamaktadır (çarpıktır).
print("\n--- Ekstra Hipotez: Shapiro-Wilk Normallik Testi ---")

# veri seti çok büyük, random 5000 seçerek testi gerçekleştiriyorum.
orneklem_fiyat = df['avg_fare'].sample(n=5000, random_state=42) if len(df) > 5000 else df['avg_fare']

stat_w, p_val_shapiro = stats.shapiro(orneklem_fiyat)

print(f"Test İstatistiği (W): {stat_w:.3f}, P-Değeri: {p_val_shapiro:.5f}")
if p_val_shapiro < 0.05:
    print("Sonuç: (H0 reddedildi): P < 0.05 olduğu için bilet fiyatları normal dağılmıyor")
    print("Yorum: Dağılım muhtemelen sağa çarpıktır (Ucuz biletler çoğunlukta, çok pahalı biletler ise kuyruğu uzatıyor).")
else:
    print("Sonuç: (H0 Reddedilemez): P > 0.05 olduğu için veri normal dağılmış.")




print("\nanalizler tamamlandı")




# =====================================================================
# AŞAMA 5: GELİŞMİŞ İSTATİSTİKSEL ANALİZ VE GÖRSELLEŞTİRME PANELİ
# =====================================================================
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

print("\n" + "="*50)
print("AŞAMA 5: İNTERAKTİF KONTROL PANELİ BAŞLATILIYOR...")
print("="*50)

def test_degisti(event):
    if test_secimi.get() == "Dağılım İncelemesi (Tek Değişken)":
        sutun2_secimi.set('') 
        sutun2_secimi.config(state="disabled") 
    else:
        sutun2_secimi.config(state="readonly")

def testi_calistir():
    test_turu = test_secimi.get()
    degisken1 = sutun1_secimi.get()
    degisken2 = sutun2_secimi.get()
    n_degeri = n_secimi.get().strip()
    
    if not test_turu or not degisken1 or not n_degeri:
        messagebox.showwarning("Uyarı", "Lütfen tüm alanları doldurunuz!")
        return
    
    try:
        # Veri Hazırlama ve n Örneklem Seçimi
        if test_turu == "Korelasyon (Sayısal vs Sayısal)":
            if not degisken2:
                messagebox.showwarning("Uyarı", "Korelasyon için 2. değişkeni seçmelisiniz!")
                return
            aktif_df = df.dropna(subset=[degisken1, degisken2])
        else:
            aktif_df = df.dropna(subset=[degisken1])

        if n_degeri.lower() != "tümü":
            n_sayisi = int(n_degeri)
            if n_sayisi < len(aktif_df):
                aktif_df = aktif_df.sample(n=n_sayisi, random_state=42)
    except Exception as e:
        messagebox.showerror("Hata", f"Veri hazırlama hatası: {e}")
        return

    # --- 1. KORELASYON ANALİZİ ---
    if test_turu == "Korelasyon (Sayısal vs Sayısal)":
        try:
            r, p_val = stats.pearsonr(aktif_df[degisken1], aktif_df[degisken2])
            yorum = "Anlamlı ilişki var." if p_val < 0.05 else "Anlamlı ilişki yok."
            
            detay = f"--- KORELASYON RAPORU (n={len(aktif_df)}) ---\n\n"
            detay += f"Değişkenler: {degisken1} & {degisken2}\n"
            detay += f"Pearson r: {r:.3f}\n"
            detay += f"P-Değeri: {p_val:.5f}\n\n"
            detay += f"SONUÇ: {yorum}"
            
            messagebox.showinfo("Korelasyon Sonucu", detay)
            
            plt.figure(figsize=(10, 6))
            sns.regplot(x=degisken1, y=degisken2, data=aktif_df, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
            plt.title(f'{degisken1} ve {degisken2} İlişkisi (r={r:.2f})')
            plt.show()
        except Exception as e:
            messagebox.showerror("Hata", f"Korelasyon hatası: {e}")

    # --- 2. DAĞILIM VE NORMALLİK ANALİZİ (KDE + Q-Q) ---
    elif test_turu == "Dağılım İncelemesi (Tek Değişken)":
        try:
            veri = aktif_df[degisken1]
            ortalama, medyan = veri.mean(), veri.median()
            std_sapma = veri.std()
            carpiklik = veri.skew()
            basiklik = veri.kurt()
            
            # Shapiro-Wilk (max 5000)
            shapiro_data = veri.sample(min(len(veri), 5000))
            w_stat, p_val = stats.shapiro(shapiro_data)
            p_yazi = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            
            # Detaylı Rapor Metni
            detay = f"--- {degisken1} DAĞILIM ANALİZİ (n={len(veri)}) ---\n\n"
            detay += f"Ortalama: {ortalama:.2f}\n"
            detay += f"Medyan: {medyan:.2f}\n"
            detay += f"Std. Sapma: {std_sapma:.2f}\n\n"
            detay += f"Çarpıklık (Skewness): {carpiklik:.2f}\n"
            detay += f"Basıklık (Kurtosis): {basiklik:.2f}\n\n"
            detay += f"Shapiro-Wilk P: {p_yazi}\n"
            detay += "Sonuç: " + ("Normal değil." if p_val < 0.05 else "Normal dağılım.")
            
            messagebox.showinfo("Detaylı Dağılım Raporu", detay)

            # Grafik Çizimi (KDE ve Q-Q yan yana değil, alt alta veya ayrı pencerelerde de olabilir ama en iyisi yan yana)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # KDE Grafiği (Sade, Dikdörtgensiz)
            sns.kdeplot(veri, fill=True, color='teal', ax=ax1, linewidth=2.5)
            ax1.axvline(ortalama, color='red', linestyle='--', label=f'Ort: {ortalama:.1f}')
            ax1.axvline(medyan, color='orange', linestyle='-', label=f'Med: {medyan:.1f}')
            ax1.set_title(f'{degisken1} Yoğunluk Eğrisi (KDE)')
            ax1.legend()

            # Q-Q Grafiği
            stats.probplot(veri, dist="norm", plot=ax2)
            ax2.get_lines()[0].set_markerfacecolor('teal')
            ax2.get_lines()[0].set_alpha(0.4)
            ax2.set_title(f'{degisken1} Q-Q Grafiği')

            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Dağılım hatası: {e}")

# --- Arayüz Tasarımı ---
arayuz = tk.Tk()
arayuz.title("Uçuş Veri Analiz Laboratuvarı")
arayuz.geometry("500x500")

sayisal_listesi = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

tk.Label(arayuz, text="Gelişmiş İstatistik Paneli", font=("Arial", 16, "bold"), fg="#333").pack(pady=20)

tk.Label(arayuz, text="1. Analiz Türü Seçin:").pack()
test_secimi = ttk.Combobox(arayuz, values=["Korelasyon (Sayısal vs Sayısal)", "Dağılım İncelemesi (Tek Değişken)"], state="readonly", width=45)
test_secimi.pack(pady=5)
test_secimi.bind("<<ComboboxSelected>>", test_degisti)

tk.Label(arayuz, text="2. Değişken 1 (X):").pack(pady=(10,0))
sutun1_secimi = ttk.Combobox(arayuz, values=sayisal_listesi, state="readonly", width=45)
sutun1_secimi.pack(pady=5)

tk.Label(arayuz, text="3. Değişken 2 (Y):").pack(pady=(10,0))
sutun2_secimi = ttk.Combobox(arayuz, values=sayisal_listesi, state="readonly", width=45)
sutun2_secimi.pack(pady=5)

tk.Label(arayuz, text="4. Örneklem Büyüklüğü (n):").pack(pady=(10,0))
n_secimi = ttk.Combobox(arayuz, values=["Tümü", "100", "500", "1000", "3000"], width=45)
n_secimi.pack(pady=5)
n_secimi.current(0)

btn = tk.Button(arayuz, text="ANALİZİ BAŞLAT", command=testi_calistir, bg="#007bff", fg="white", font=("Arial", 11, "bold"), height=2, width=20)
btn.pack(pady=30)

arayuz.attributes('-topmost', True)
arayuz.mainloop()