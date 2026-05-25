# Real Estate ML Tree & KD-Tree Scraper (Explainable PropTech)

Bu proje, emlak fiyat tahmini ve arama özelliklerini modern bir UI ile sunan uçtan uca (end-to-end) bir "Açıklanabilir PropTech" sistemidir. Üniversite jürisi için hazırlanmıştır.

## Özellikler
- **Mekânsal Arama (Spatial Search):** Özel yapım KD-Tree veri yapısı kullanılarak `O(√N + k)` karmaşıklığında hızlı arama.
- **Açıklanabilir Fiyat Tahmini (XAI):** Global Random Forest modeli ile lokal Gaussian RBF Kernel kombinasyonu sayesinde, tahminlerin yanında benzer mülkleri (comparables) ve modele ait güven skorunu gösterir.
- **Canlı Veri Kazıma (Live Scraper):** Asenkron Craigslist scraper'ı, Node (Bağlı Liste) tabanlı FIFO Queue yapısıyla çalışır. Toplu (batch) olarak `O(k log N)` hızında KD-Tree'ye aktarır. LIFO Stack yapısıyla son işlemleri geri alma (undo) özelliği vardır.
- **Modern UI:** Karanlık/Aydınlık mod destekli, interaktif Leaflet.js tabanlı arayüz.

## Ekip İçin Hızlı Başlangıç (Tek Tıkla Çalıştırma)

Projede veritabanı (Supabase) bağlantılarının bulunduğu `.env` dosyası repoya dahil edilmiştir. Docker kurulu olan herhangi bir bilgisayarda projeyi ayağa kaldırmak için tek bir dosya çalıştırmanız yeterlidir.

### Windows Kullanıcıları İçin
Proje dizinindeki `start.bat` dosyasına çift tıklayarak çalıştırın veya terminalden şu komutu girin:
```cmd
start.bat
```

### macOS / Linux Kullanıcıları İçin
Terminalden aşağıdaki komutu girerek projeyi başlatabilirsiniz:
```bash
./start.sh
```

### Uygulamaya Erişim
Docker indirme ve derleme işlemlerini bitirip uygulamayı başlattıktan sonra:
- **Kullanıcı Arayüzü (UI):** [http://localhost:8080/](http://localhost:8080/)
- **API Dokümantasyonu (Swagger):** [http://localhost:8080/docs](http://localhost:8080/docs)
- **API Sağlık Durumu:** [http://localhost:8080/health](http://localhost:8080/health)

*(Not: İlk çalıştırmada, uygulama veritabanından yaklaşık 22.000 satır veriyi çekip bellekte KD-Tree inşa edecek ve Random Forest modelini eğitecektir. Bu nedenle uygulamanın hazır olması 30-90 saniye sürebilir.)*

## Mimari Dokümantasyon
Projenin algoritmaları, veri yapıları karmaşıklık analizleri, formülleri ve sistem darboğaz analizleri ile ilgili detaylı ve akademik bilgi için lütfen [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) dosyasını okuyun.
