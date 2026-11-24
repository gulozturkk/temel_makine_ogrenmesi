# ✍️ El Yazısı Rakam Tanıma (Handwritten Digit Recognition with CNN)

Bu proje, banka çekleri ve resmi evraklar üzerindeki el yazısı rakamların dijital ortama aktarılması sürecini otomatikleştirmek amacıyla geliştirilmiş bir **Evrişimli Sinir Ağı (CNN)** tabanlı derin öğrenme modelidir[cite: 9].

---

## 🎯 Problem Tanımı ve Önemi
* **Problem:** El yazısı rakamların manuel olarak okunup dijital ortama aktarılması zaman alıcı, yorucu ve insan hatasına açık bir süreçtir[cite: 9].
* **Çözüm ve Amaç:** $28 \times 28$ piksel boyutundaki el yazısı rakam görsellerini yüksek doğrulukla otomatik olarak tanıyan ve 0–9 arasındaki 10 sınıfa ayıran bir CNN modeli geliştirilmiştir[cite: 9]. Bu otomasyon iş yükünü azaltır, hata oranını en aza indirir ve operasyonel verimliliği artırır[cite: 9].

---

## 📁 Veri Seti ve Ön İşleme
* **Veri Seti:** `tf.keras.datasets.mnist`[cite: 9]
* **Örnek Sayısı:** Toplam **70.000** adet gri tonlamalı (grayscale) el yazısı rakam görüntüsü (**60.000** eğitim, **10.000** test)[cite: 9].
* **Girdi ve Sınıflar:** Her örnek $28 \times 28$ piksellik matrislerden oluşur ve 0 ile 9 arasındaki **10 dengeli sınıfı** temsil eder (eğitim setinde sınıf başına yaklaşık 6.000 örnek)[cite: 9].
* **Uygulanan Ön İşleme Adımları:**
  1. **Normalizasyon:** `[0, 255]` aralığındaki piksel değerleri `255.0`'a bölünerek `[0, 1]` aralığına ölçeklendirilmiştir[cite: 10].
  2. **Kanal Boyutlandırma (Reshape):** İki boyutlu `(28, 28)` matrisler, `Conv2D` katmanının standart giriş formatına uygun olacak şekilde tek kanallı 3 boyutlu `(28, 28, 1)` tensörlere dönüştürülmüştür[cite: 10, 11].

---

## 🔍 Keşifçi Veri Analizi (EDA) ve Görselleştirme
Proje kapsamında veri setini ve model davranışını anlamak için 3 temel görselleştirme gerçekleştirilmiştir:
1. **Örnek El Yazısı Rakamlar ($4 \times 4$ Izgara):** Farklı kişilerin aynı rakamları farklı eğim, kalınlık ve yazım stilleriyle yazdığı gözlemlenerek gerçek dünya varyasyonları incelenmiştir[cite: 9].
2. **Eğitim ve Doğrulama Eğrileri (Accuracy & Loss):** 10 epoch boyunca eğitim kaybının istikrarlı biçimde düştüğü, eğitim ve doğrulama eğrilerinin birbirine paralel ilerleyerek aşırı öğrenme (overfitting) olmadan %99 bandına ulaştığı doğrulanmıştır[cite: 10].
3. **Hata Matrisi (Confusion Matrix):** `seaborn` ısı haritası (heatmap) ile test setindeki gerçek ve tahmin edilen sınıfların dağılımı analiz edilmiştir[cite: 10].

---

## 🧠 Model Mimarisi ve Eğitim Parametreleri

Uzamsal piksel ilişkilerini en iyi yakalayan yapı olduğu için **Convolutional Neural Network (CNN)** mimarisi tercih edilmiştir[cite: 11].

### Katman Yapısı
| Katman | Detaylar | Amaç |
| :--- | :--- | :--- |
| **Conv2D** | 32 filtre, $3 \times 3$ çekirdek, `ReLU` | Temel kenar ve çizgi özelliklerini çıkarma[cite: 11] |
| **MaxPooling2D** | $2 \times 2$ havuzlama | Öznitelik haritası boyutunu yarıya indirerek hesaplama yükünü azaltma[cite: 11] |
| **Conv2D** | 64 filtre, $3 \times 3$ çekirdek, `ReLU` | Daha derin ve karmaşık rakam örüntülerini öğrenme[cite: 11] |
| **Flatten** | Tek boyutlu vektör | Evrişim matrislerini tam bağlantılı katmanlara aktarma[cite: 11] |
| **Dense + Dropout** | 128 nöron (`ReLU`) + `Dropout(0.3)` | Özellikleri birleştirme ve rastgele nöron kapatarak ezberlemeyi (overfitting) önleme[cite: 11] |
| **Çıktı Katmanı (Dense)** | 10 nöron, `Softmax` | 10 sınıf (0–9) için olasılık dağılımı üretme[cite: 11] |

### Eğitim Konfigürasyonu
* **Loss Fonksiyonu:** `sparse_categorical_crossentropy`[cite: 12]
* **Optimizer:** `Adam`[cite: 12]
* **Epoch Sayısı:** `10`[cite: 12]
* **Batch Size:** `64`[cite: 12]
* **Validation Split:** Eğitim verisinin **%20**'si doğrulama seti olarak ayrılmıştır[cite: 12].

---

## 📈 Değerlendirme Sonuçları ve Hata Analizi
* **Test Doğruluğu (Accuracy):** 10.000 test görüntüsü üzerinde **%99.12** doğruluk elde edilmiştir[cite: 12].
* **Precision, Recall ve F1-Score:** Tüm sınıflarda ortalama **0.99** seviyesinde dengeli bir performans yakalanmıştır[cite: 13].
* **Güçlü Yönler:** Hafif eğrilik, kalınlık farkları ve yazım varyasyonlarına karşı yüksek genelleme yeteneği[cite: 13].
* **Zayıf Yönler:** Hata matrisi incelemesinde nadir hataların genellikle geometrik olarak birbirine benzeyen rakam çiftlerinde (örneğin üstü açık yazılmış `3` ile `8` veya dik hatları benzeyen `4` ile `9`) oluştuğu tespit edilmiştir[cite: 13].

---

## 🚀 Gelecek Geliştirmeler
* **Veri Artırma (Data Augmentation):** Eğitim görsellerine hafif döndürme, kaydırma ve yakınlaştırma uygulayarak farklı el yazısı stillerine karşı dayanıklılığı artırmak[cite: 13].
* **Mimari İyileştirme:** Ek `BatchNormalization` veya `Dropout` katmanlarıyla ağı derinleştirmek[cite: 13].
* **Hiperparametre Optimizasyonu:** Epoch sayısını ve learning rate değerini optimize ederek test başarımını %99.5'in üzerine çıkarmak[cite: 13].

---

## 🛠️ Kurulum ve Çalıştırma

1. Repoyu klonlayın:
   git clone [https://github.com/gulozturkk/repo-adi.git](https://github.com/gulozturkk/repo-adi.git)
   cd repo-adi

2.Gerekli kütüphaneleri yükleyin:
  pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter

3.Notebook dosyasını açıp çalıştırın:
  jupyter notebook hafta_2_proje.ipynb
