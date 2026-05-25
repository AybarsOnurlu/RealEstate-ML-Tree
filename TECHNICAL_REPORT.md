# Explainable PropTech — Mimari Teknik Rapor

**Proje:** Real Estate Search & Price Prediction API  
**Versiyon:** 2.0.0  
**Tarih:** 25 Mayıs 2026  
**Hazırlayan:** Aybars Onurlu  
**Veri Seti:** King County House Sales (~22.006 kayıt)  
**Çalışma Ortamı:** Docker (linux/amd64), AWS EC2 t3.large, Uvicorn/FastAPI

---

## İçindekiler

1. [Executive Summary](#1-executive-summary)
2. [Sistem Mimarisi Genel Bakış](#2-sistem-mimarisi-genel-bakış)
3. [Veri Yapıları (Data Structures)](#3-veri-yapıları-data-structures)
4. [Makine Öğrenmesi & XAI Mimarisi](#4-makine-öğrenmesi--xai-mimarisi)
5. [Ölçeklenebilirlik & Performans](#5-ölçeklenebilirlik--performans)
6. [Frontend & UI Mantığı](#6-frontend--ui-mantığı)
7. [API Yüzey Alanı (Endpoint Kataloğu)](#7-api-yüzey-alanı-endpoint-kataloğu)
8. [Sonuç & Gelecek Çalışmalar](#8-sonuç--gelecek-çalışmalar)

---

## 1. Executive Summary

Bu proje, "Açıklanabilir Emlak Fiyat Tahmini" (Explainable PropTech) ihtiyacını karşılamak üzere tasarlanmış bir uçtan uca (end-to-end) yazılım sistemidir. Sistem, aşağıdaki üç temel gereksinimi eş zamanlı olarak yerine getirir:

| Gereksinim | Çözüm | Kanıtlama Yöntemi |
|---|---|---|
| **Near Real-Time Mekânsal Sorgulama** | Özel yazılmış 2D KD-Tree (`kdtree.py`) ile Haversine mesafe ölçütüne dayalı O(√N + k) radius search ve O(k log k · log N) KNN sorgusu | `/benchmark` endpointi ile KD-Tree vs. Brute-Force karşılaştırmalı hız testi |
| **Açıklanabilir Fiyat Tahmini (XAI)** | Random Forest global tahmin + Gaussian RBF Kernel tabanlı lokal ağırlıklı regresyon hibrit modeli; dinamik α katsayısı ile harmanlanmış nihai tahmin | `/predict` yanıtında `rbf_weight` (güven skoru) ve `comparable_properties` (XAI) dizisi |
| **Canlı Veri Akışı (Live Ingestion)** | `httpx.AsyncClient` tabanlı asenkron Craigslist scraper → `PropertyQueue` (FIFO) → toplu DB ekleme → `ScraperUndoStack` (LIFO) ile geri alma | Event loop bloklanma analizi ve `stress_test.py` ile eş-zamanlı yük testi |

Tüm bileşenler tek bir Docker konteyneri içinde FastAPI/Uvicorn üzerinde çalışır; frontend statik dosyalar olarak aynı süreç tarafından sunulur (`app.mount("/")`). Dış bağımlılık olarak yalnızca Supabase (PostgreSQL) veritabanı bulunmaktadır.

---

## 2. Sistem Mimarisi Genel Bakış

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         🖥️ Frontend (Browser)                             │
│   index.html + app.js + style.css   │   Leaflet.js Harita (CartoDB)       │
└──────────────┬─────────────────────────────────────────┬──────────────────┘
               │ fetch() / POST                          │
               ▼                                         │
┌──────────────────────────────────────────────────────────────────────────┐
│                    ⚡ FastAPI / Uvicorn (Port 8080)                      │
│                                                                          │
│  /health  /search  /knn  /predict  /metrics  /sync  /benchmark          │
│  /scrape/trigger   /scrape/process   /scrape/undo                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    🧠 In-Memory Engine                           │    │
│  │   KD-Tree (22.006 düğüm)  │  Random Forest (150 ağaç)          │    │
│  │   feature_scales (σ cache) │  model_metrics (R², RMSE, MAE)     │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    📦 Veri Yapıları                               │    │
│  │   PropertyQueue (Linked List FIFO)                               │    │
│  │   ScraperUndoStack (Linked List LIFO)                            │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└───────────────────┬────────────────────────────┬─────────────────────────┘
                    │                            │
                    ▼                            ▼
          ┌─────────────────┐          ┌──────────────────┐
          │  ☁️ Supabase     │          │  🌐 Craigslist    │
          │  (PostgreSQL)   │          │  Seattle          │
          └─────────────────┘          └──────────────────┘
```

### Başlatma Yaşam Döngüsü (Lifespan)

Uygulama başlatıldığında `lifespan()` fonksiyonu (app.py satır 244–254) devreye girer:

1. `fetch_all_properties()` → Supabase'den 22.006 kaydı sayfalayarak (`page_size=1000`) çeker
2. `rebuild_tree_and_retrain(props)` → KD-Tree dengeli inşa + Random Forest eğitimi + `feature_scales` önbellek hesaplaması
3. Tüm bu süreç tamamlanmadan hiçbir endpoint istek kabul etmez (`_require_tree()`, `_require_model()` koruma fonksiyonları)

---

## 3. Veri Yapıları (Data Structures)

### 3.1 KD-Tree — Özel 2D Mekânsal İndeks

> **Kaynak Dosya:** `kdtree.py` (289 satır, sıfırdan yazılmış, hiçbir harici mekânsal kütüphane kullanılmamıştır)

#### 3.1.1 Veri Modeli

```python
@dataclass
class Property:           # kdtree.py satır 23–35
    id: str
    price: float
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    yr_built: float
    lat: float            # Bölme ekseni 0
    long: float           # Bölme ekseni 1
```

Her düğüm (`KDNode`) bir `Property` referansı, bir `axis` (0=lat, 1=long) değeri ve sol/sağ çocuk işaretçileri taşır. Bellek verimliliği için `__slots__` kullanılmıştır (kdtree.py satır 45).

#### 3.1.2 İnşa Algoritması — `build()`

**Karmaşıklık:** O(N log²N)

(kdtree.py satır 113–136)

Dengeli bir ağaç oluşturmak için **medyan bölme (median-of-sorted-list)** stratejisi uygulanır:

1. Her özyineleme derinliğinde `axis = depth % 2` ile bölme ekseni belirlenir
2. Liste, mevcut eksene göre sıralanır (`props.sort(key=...)`)
3. Ortanca eleman (`props[len(props)//2]`) pivot olarak seçilir
4. Sol alt ağaç `props[:median]`, sağ alt ağaç `props[median+1:]` ile özyinelemeli olarak inşa edilir

22.006 kayıt için oluşan ağacın yüksekliği: **15** (ideal denge: ⌈log₂(22006)⌉ = 15)

#### 3.1.3 Dinamik Ekleme — `insert()`

**Karmaşıklık:** Ortalama O(log N), En kötü O(N) (dengesiz ağaç)

(kdtree.py satır 142–158)

Scraper tarafından yeni eklenen kayıtlar, ağacı sıfırdan kurmak yerine mevcut yapıya doğrudan eklenir. Bu, `process_queue()` fonksiyonunda toplu (batch) kullanım için kritik öneme sahiptir:

```python
for row in inserted_rows:        # app.py satır 806–821
    p = Property(...)
    kd_tree.insert(p)             # O(log N) — mevcut ağacı koruyarak ekle
```

`k` adet yeni kayıt için toplam maliyet: **O(k · log N)** — Tam rebuild olan O(N log²N) ile karşılaştırıldığında büyük ölçekli veri setlerinde anlamlı bir tasarruf sağlar.

#### 3.1.4 Mesafe Ölçütü — Haversine Formülü

(kdtree.py satır 81–98)

Dünya yüzeyindeki iki coğrafi nokta arasındaki büyük daire mesafesini (km) hesaplar:

```
d = 2R · atan2(√a, √(1−a))
```

Burada:

```
a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
```

R = 6371.0 km (WGS-84 ortalaması). Hiçbir harici kütüphane kullanılmamış, tamamen `math` modülü ile saf Python'da yazılmıştır.

#### 3.1.5 Radius Search — `search_within_radius()`

**Karmaşıklık:** O(√N + k) [düzgün dağılım altında]

(kdtree.py satır 164–216)

**Budama (Pruning) Stratejisi:**

Her düğümde, sorgu noktasından bölme hiperdüzlemine olan minimum Haversine mesafesi bir **düzlemsel yaklaşım** ile hesaplanır:

- **Eksen 0 (enlem):** `diff_km = |Δlat| × 111.32`
- **Eksen 1 (boylam):** `diff_km = |Δlon| × 111.32 × cos(lat)`

Eğer bu mesafe arama yarıçapını aşıyorsa, uzak alt ağacın tamamı budanır — bu da lineer taramaya kıyasla dramatik bir hızlanma sağlar.

#### 3.1.6 k-Nearest Neighbors — `k_nearest_neighbors()`

**Karmaşıklık:** O(k log k · log N)

(kdtree.py satır 222–273)

Python'un `heapq` modülü bir **min-heap** olduğundan, mesafe değerleri negatif çevrilerek `(-dist, id(node), property)` üçlüsü ile bir **max-heap** simüle edilir. Bu yapı sayesinde:

1. Heap boyutu `k`'yı aşmaz → bellek sabit
2. Yeni bir aday bulunduğunda yalnızca heap'teki **en uzak** eleman değiştirilir (`heapreplace`)
3. Uzak alt ağaç yalnızca, heap'teki en kötü mesafenin bölme düzlemi mesafesinden büyük olduğu durumlarda keşfedilir

---

### 3.2 PropertyQueue — FIFO Kuyruk (Linked List)

> **Kaynak Dosya:** `structures.py` satır 17–58

| Operasyon | Karmaşıklık | Kullanım Yeri |
|---|---|---|
| `enqueue(item)` | O(1) | `trigger_scrape()` — her çekilen ilan için (app.py satır 725) |
| `dequeue()` | O(1) | `process_queue()` — toplu boşaltma (app.py satır 777–782) |
| `is_empty()` | O(1) | Döngü kontrol koşulu |
| `size()` | O(1) | Frontend'e kuyruk boyutu bildirimi |
| `peek()` | O(1) | Kuyruğun başındaki elemanı okuma |

**Neden Python `list` veya `collections.deque` kullanılmadı?**

Proje, veri yapıları dersinin uygulamalı bir çıktısı olarak, bağlı düğüm (linked node) tabanlı saf implementasyonları gerekli kılmaktadır. `Node` sınıfı `__slots__` ile optimize edilmiştir; `head` ve `tail` işaretçileri ile hem ekleme hem çıkarma O(1) garanti edilir.

**Veri Akış Senaryosu:**

```
Craigslist → [await client.get()] → BeautifulSoup parse → enqueue(prop)
                                                             │
                     process_queue() → while dequeue() → batch[] → Supabase INSERT
```

---

### 3.3 ScraperUndoStack — LIFO Yığın (Linked List)

> **Kaynak Dosya:** `structures.py` satır 60–93

| Operasyon | Karmaşıklık | Kullanım Yeri |
|---|---|---|
| `push(item)` | O(1) | `process_queue()` — başarılı insert sonrası batch ID'lerini kaydeder (app.py satır 799) |
| `pop()` | O(1) | `undo_scrape()` — son batch'in ID'lerini geri alır (app.py satır 865) |
| `is_empty()` | O(1) | Geri alma olasılığı kontrolü |

**Transaksiyonel Garanti:**

Stack, her başarılı `Supabase.insert()` sonrası eklenen satırların UUID listesini (`inserted_ids`) saklar. `undo_scrape()` çağrıldığında:

1. `pop()` ile son batch'in ID'leri alınır
2. `supabase.table("properties").delete().in_("id", last_batch_ids)` ile veritabanından silinir
3. `rebuild_tree_and_retrain()` ile ağaç ve model güncellenir

Frontend tarafında da bu LIFO davranışı yansıtılır: `liveScrapedMarkerStacks` dizisinin son elemanı (`pop()`) haritadan kaldırılır (app.js satır 481–484).

---

## 4. Makine Öğrenmesi & XAI Mimarisi

### 4.1 Global Model — Random Forest Regressor

(app.py satır 196–237)

| Hiperparametre | Değer | Gerekçe |
|---|---|---|
| `n_estimators` | 150 | Yeterli ağaç sayısı ile varyans düşürme |
| `max_depth` | 20 | Aşırı öğrenmeyi sınırla |
| `min_samples_split` | 10 | Yaprak düğüm genellemesi |
| `min_samples_leaf` | 4 | Gürültü direnci |
| `oob_score` | True | Out-of-bag hata tahmini (cross-validation alternatifi) |
| `n_jobs` | -1 | Tüm CPU çekirdeklerinde paralel eğitim |

**Özellikler (Features):** `bedrooms`, `bathrooms`, `sqft_living`, `yr_built`, `lat`, `long`  
**Hedef (Target):** `price`  
**Veri Bölme:** `train_test_split(test_size=0.2, random_state=42)`

Eğitim tamamlandığında `model_metrics` sözlüğü R², RMSE, MAE, OOB skoru ve `feature_importances_` değerlerini saklar. Bu metrikler `/metrics` endpointi üzerinden frontend'e sunulur.

---

### 4.2 Lokal Model — Geographically Weighted Regression (Gaussian RBF Kernel)

(app.py satır 448–524)

Bu katman, `/predict` endpointinde Random Forest'ın global tahminini mekânsal ve özellik bazlı komşuluk bilgisiyle zenginleştirir. Dört aşamadan oluşur:

#### Aşama A — Mekânsal Filtreleme (Spatial Filtering)

```python
neighbors = kd_tree.k_nearest_neighbors(request.lat, request.long, 30)
# O(k log k · log N) — 30 en yakın coğrafi komşu
```

Kullanıcının seçtiği koordinata en yakın **30 komşu** KD-Tree'den çekilir. Bu, lineer tarama yerine O(30 · log 30 · log 22006) ≈ O(30 × 5 × 15) = ~2250 operasyonla tamamlanır.

#### Aşama B — Mekânsal Ağırlık (W_spatial) — Gaussian Penalty

```
W_spatial = exp(−d_spatial² / (2 · γ_spatial²))     γ_spatial = 2.0 km
```

(app.py satır 467)

Bu formül, fiziksel mesafeye Gaussian bozunma uygular:

| Mesafe | W_spatial |
|---|---|
| 0 km | 1.000 |
| 1 km | 0.882 |
| 2 km (= γ) | 0.607 |
| 4 km | 0.135 |
| 6 km | 0.011 |

2 km ötesindeki komşuların etkisi hızla sıfıra yaklaşır.

#### Aşama C — Normalize Özellik Benzerliği (W_feature) — RBF Kernel

```python
sigma_beds  = feature_scales.get("bedrooms", 1.0)     # Önbellekten O(1)
sigma_baths = feature_scales.get("bathrooms", 1.0)
sigma_sqft  = feature_scales.get("sqft_living", 1.0)
sigma_year  = feature_scales.get("yr_built", 1.0)
```

Her özellik, **eğitim zamanında bir kez hesaplanan** standart sapma (σ) ile normalize edilir (app.py satır 230–232). Bu, birimler arası ölçek farklılığını ortadan kaldırır (örn. `sqft_living` ~800–5000 aralığında iken `bedrooms` 1–6 aralığındadır).

```
Δᵢ = |x_kullanıcı,ᵢ − x_komşu,ᵢ| / σᵢ

FeatDist = Δ_beds + Δ_baths + Δ_sqft + Δ_year

W_feature = exp(−FeatDist² / (2 · γ_feature²))     γ_feature = 1.0
```

(app.py satır 469–476)

#### Aşama D — Birleşik Ağırlık ve Dinamik Alfa Harmanlaması

```python
W_total = W_spatial × W_feature                      # app.py satır 479

local_pred = Σ(W_total_i × price_i) / Σ(W_total_i)  # Ağırlıklı ortalama (satır 490)

alpha = min(0.7, Σ(W_total) / 10.0)                  # Dinamik güven katsayısı (satır 494)

predicted = α × local_pred + (1 − α) × rf_pred       # Nihai harmanlanmış tahmin (satır 497)
```

**Dinamik α'nın Önemi:**

- Toplam ağırlık (`Σ W_total`) yüksekse → birçok benzer komşu var → α artırılır (lokal bilgiye güvenilir)
- Toplam ağırlık düşükse → kullanıcı bir "outlier" kombinasyonu girmiş → α düşük kalır (RF global tahminine güvenilir)
- α hiçbir zaman **0.7'yi aşmaz** → global model her zaman minimum %30 ağırlık taşır

#### XAI Çıktısı — Karşılaştırılabilir Mülkler (Comparables)

(app.py satır 500–515)

En yüksek `W_total` ağırlığına sahip ve `W_total > 0.05` eşiğini aşan ilk **5 komşu**, `comparable_properties` dizisi olarak döndürülür. Bu, **Explainable AI (XAI)** prensibine uygun olarak kullanıcıya "Neden bu fiyat tahmin edildi?" sorusunun yanıtını sunar.

---

## 5. Ölçeklenebilirlik & Performans

### 5.1 Asenkron I/O — Event Loop Koruma

**Sorun (v1.x):**

Orijinal `trigger_scrape()` fonksiyonu senkron `httpx.get()` kullanıyordu. Her Craigslist isteği (liste sayfası + ~15 detay sayfası) toplam **8–45 saniye** boyunca Uvicorn'un async event loop'unu tamamen bloke ediyordu. Bu süre zarfında `/predict`, `/search`, `/knn` gibi hiçbir endpoint yanıt veremiyordu.

**Çözüm (v2.0):**

(app.py satır 635–753)

```python
async def trigger_scrape():   # ← def → async def
    async with httpx.AsyncClient(
        headers=HEADERS,
        follow_redirects=True,
        timeout=httpx.Timeout(connect=8.0, read=15.0, write=5.0, pool=5.0),
    ) as client:
        resp = await client.get(CL_LIST_URL)       # ← Non-blocking
        ...
        dr = await client.get(link)                 # ← Non-blocking
```

**Etkisi:**

- Her `await` noktasında event loop kontrolü geri alır
- Scraper çalışırken `/predict` ve `/search` istekleri paralel olarak işlenebilir
- Tek bir `AsyncClient` örneği bağlantı havuzunu (connection pool) verimli kullanır
- Granüler timeout yapılandırması (`connect=8s`, `read=15s`) ile bağlantı çökmelerine karşı dayanıklılık

### 5.2 Batch Insert & İnkremental KD-Tree Güncelleme

**Sorun (v1.x):**

`process_queue()` her çağrıldığında:

1. ✅ Kuyruktan tüm elemanları çekiyor
2. ✅ Tek bir Supabase INSERT yapıyor
3. ❌ `fetch_all_properties()` ile 22.000 kaydı DB'den tekrar çekiyor
4. ❌ `rebuild_tree_and_retrain()` ile ağacı O(N log²N) maliyetle sıfırdan kuruyordu

**Çözüm (v2.0) — 4 Aşamalı Pipeline:**

(app.py satır 760–852)

```
Aşama 1:  Kuyruğu tamamen boşalt           → O(k)
Aşama 2:  TEK Supabase INSERT              → 1 HTTP çağrısı
Aşama 3:  kd_tree.insert(p) × k            → O(k · log N)  ← Kritik iyileştirme
Aşama 4:  RF yeniden eğit (1 kez)          → O(N) veri çekimi + O(N · T) eğitim
```

**Karmaşıklık Karşılaştırması (k=15, N=22.006):**

| | Eski (Tam Rebuild) | Yeni (İnkremental) |
|---|---|---|
| DB Round-Trip | 22.006 satır çek | 15 satır (yalnızca yeni) |
| KD-Tree İnşa | O(22.006 · log²22.006) ≈ 4.9M op | O(15 · log 22.006) ≈ 225 op |
| RF Eğitim | 1 kez | 1 kez |
| Toplam Kazanç | — | **~21.000× daha az ağaç operasyonu** |

### 5.3 O(1) Standart Sapma Önbelleği (`feature_scales`)

(app.py satır 73, satır 229–232)

Her `/predict` çağrısında 22.006 kayıt üzerinden σ hesaplamak O(N) maliyetine sahiptir. Bunun yerine:

```python
feature_scales: Dict[str, float] = {}  # Modül seviyesinde global sözlük

# rebuild_tree_and_retrain() içinde — eğitim zamanında BİR KEZ hesaplanır:
for f in ["bedrooms", "bathrooms", "sqft_living", "yr_built"]:
    std_val = float(np.std([getattr(p, f) for p in properties]))
    feature_scales[f] = std_val if std_val > 0.0 else 1.0
```

Her `/predict` çağrısında erişim:

```python
sigma_beds = feature_scales.get("bedrooms", 1.0)  # O(1) dict lookup
```

Bu optimizasyon, 100 eş-zamanlı `/predict` isteğinde **100 × 22.006 = 2.2M** gereksiz operasyonu ortadan kaldırır.

### 5.4 Stres Testi Altyapısı

`stress_test.py` — Projede yer alan kapsamlı performans test scripti aşağıdaki senaryoları kapsar:

| Senaryo | Eş-zamanlı İstek | Ölçülen Metrikler |
|---|---|---|
| `/search` (küçük/büyük radius) | 75 × 3 tur | Ortalama, P95, P99 gecikme (ms) |
| `/knn` (k=5..50) | 75 × 3 tur | Bellek değişimi (ΔMB) |
| `/predict` (Gaussian RBF + RF) | 75 × 3 tur | Timeout/deadlock sayısı |
| `/scrape/trigger` | 8 (rate limit) | Craigslist I/O gecikmesi |
| Karma yük (mixed) | 150 | Throughput (req/s) |

---

## 6. Frontend & UI Mantığı

> **Kaynak Dosyalar:** `index.html` (134 satır), `app.js` (502 satır), `style.css` (524 satır)

### 6.1 Mimari Kararlar

| Karar | Uygulama |
|---|---|
| **API Base URL** | `window.location.origin` → port/sunucu bağımsız (app.js satır 2) |
| **Harita Kütüphanesi** | Leaflet.js v1.9.4 + CartoDB Positron/Dark tile katmanları |
| **Stil Yaklaşımı** | CSS Custom Properties (`--primary`, `--bg-glass` vb.) ile dark/light mode |
| **Animasyon** | `slideIn`, `fadeIn`, `popIn`, `pulse` keyframe animasyonları |

### 6.2 XAI Görselleştirme — Comparable Polylines

(app.js satır 309–358)

Tahmin yanıtında dönen `comparable_properties` dizisi, haritada şu şekilde görselleştirilir:

1. **Kesikli Polyline:** Kullanıcının seçtiği noktadan her karşılaştırılabilir mülke kesikli mor çizgi (`dashArray: '5, 5'`, `color: '#6366f1'`)
2. **Comparable Marker:** Her mülk için mor dolu daire (`fillColor: '#6366f1'`)
3. **Popup Detay:** Comp #, fiyat, yatak, banyo, metrekare bilgisi
4. **Otomatik Zoom:** `L.featureGroup([...compMarkers, currentClickMarker]).getBounds().pad(0.1)` ile tüm comparables ve hedef noktayı kapsayan zoom

**Garbage Collection — Önceki Çizimler:**

Her yeni tahmin yapılmadan önce önceki polyline ve marker'lar temizlenir (app.js satır 272–275):

```javascript
compLines.forEach(line => map.removeLayer(line));
compMarkers.forEach(m => map.removeLayer(m));
compLines = [];
compMarkers = [];
```

### 6.3 AI Confidence Badge (Blend Alpha Göstergesi)

(app.js satır 280–302)

API'den dönen `rbf_weight` (dinamik α) değeri, görsel bir güven çubuğuna dönüştürülür:

```javascript
const percent = Math.round(data.rbf_weight * 100);
blendBarFill.style.width = `${percent}%`;

if (percent > 70)       → Yeşil (#10b981)  → "High Confidence"
else if (percent > 40)  → Sarı (#f59e0b)   → "Medium Confidence"
else                    → Kırmızı (#ef4444) → "Low Confidence"
```

Bu gösterge, kullanıcıya **"Bu tahminin ne kadarı mekânsal komşuluk verisine, ne kadarı global makine öğrenmesi modeline dayandığını"** açıklar — Explainable AI'ın kullanıcı arayüzü katmanındaki somut yansımasıdır.

### 6.4 Scraper Marker Garbage Collection

(app.js satır 452–458)

Canlı scraper tekrar tekrar tetiklenip işlendiğinde, haritada birikerek tarayıcı performansını düşürebilecek binlerce marker oluşabilir. Bu sorun şöyle çözülür:

```javascript
let totalScraped = liveScrapedMarkerStacks.reduce((sum, batch) => sum + batch.length, 0);
while (totalScraped > 1000 && liveScrapedMarkerStacks.length > 0) {
    let oldestBatch = liveScrapedMarkerStacks.shift();   // FIFO: en eski batch
    oldestBatch.forEach(m => map.removeLayer(m));         // DOM'dan temizle
    totalScraped -= oldestBatch.length;
}
```

Bu mekanizma, toplam marker sayısını **1000 ile sınırlar** ve en eski batch'leri otomatik olarak temizler — sürekli scrape döngülerinde tarayıcı bellek tüketimini sabit tutar.

### 6.5 Backend–Frontend Stack Simetrisi

Frontend'deki `liveScrapedMarkerStacks` dizisi, backend'deki `ScraperUndoStack` ile birebir LIFO simetrisi korur:

```
Backend:  undo_stack.pop()                → Son batch ID'leri alınır, DB'den silinir
Frontend: liveScrapedMarkerStacks.pop()   → Son marker grubu haritadan kaldırılır
```

Bu simetri, kullanıcı "Undo Last Scrape" butonuna bastığında hem veritabanı hem de harita durumunun tutarlı kalmasını garanti eder.

### 6.6 Canlı Scraped Marker Animasyonu

Canlı veri ile eklenen marker'lar, `.live-marker` CSS sınıfı ile **pulse animasyonu** alır (style.css satır 409–418):

```css
@keyframes pulse {
    0%   { stroke-width: 2px; stroke-opacity: 1;   }
    50%  { stroke-width: 8px; stroke-opacity: 0.3; }
    100% { stroke-width: 2px; stroke-opacity: 1;   }
}
```

Bu animasyon, canlı veriden gelen kayıtları mevcut (statik) veri setinden görsel olarak ayırt etmeyi sağlar.

---

## 7. API Yüzey Alanı (Endpoint Kataloğu)

| HTTP | Path | Etiket | Açıklama | Karmaşıklık |
|---|---|---|---|---|
| GET | `/health` | System | KD-Tree boyutu, yüksekliği, model durumu, R² | O(1) |
| GET | `/search` | Spatial Search | Haversine radius search | O(√N + k) |
| GET | `/knn` | Spatial Search | k-Nearest Neighbors | O(k log k · log N) |
| POST | `/predict` | ML Prediction | RF + Gaussian RBF hibrit tahmin | O(k log k · log N) + O(30) |
| GET | `/metrics` | ML Prediction | R², RMSE, MAE, OOB, feature importance | O(1) |
| POST | `/sync` | System | DB → KD-Tree rebuild + RF retrain | O(N log²N) |
| GET | `/benchmark` | System | KD-Tree vs. Brute-Force hız karşılaştırması | O(√N + k) + O(N) |
| GET | `/scrape/trigger` | Live Scraper | Async Craigslist scrape → Queue | I/O bound |
| POST | `/scrape/process` | Live Scraper | Queue → Batch INSERT → Tree insert → RF retrain | O(k log N) + O(N·T) |
| POST | `/scrape/undo` | Live Scraper | Stack pop → DB delete → Full rebuild | O(N log²N) |

---

## 8. Sonuç & Gelecek Çalışmalar

### Ulaşılan Durum

Sistem, sıfırdan yazılmış özel veri yapıları (KD-Tree, Queue, Stack), Gaussian RBF Kernel tabanlı Açıklanabilir Yapay Zeka tahmini ve asenkron canlı veri toplama kabiliyetini tek bir mikro servis mimarisinde birleştirmektedir.

- 22.006 kayıt üzerinde dengeli KD-Tree yüksekliği **15**'e sabitlenmiştir
- Radius search ve KNN sorguları **milisaniye düzeyinde** yanıt vermektedir
- Asenkron I/O geçişi ile event loop bloklanması **ortadan kaldırılmıştır**
- Toplu (batch) insert stratejisi ile artımlı güncelleme maliyeti **logaritmik seviyelere** düşürülmüştür
- XAI bileşeni, her tahmin için comparable mülkler ve güven skoru sunarak **model kararlarını şeffaf** kılmaktadır

### Gelecek Çalışmalar

| Konu | Açıklama | Öncelik |
|---|---|---|
| `asyncio.Lock` | `rebuild_tree_and_retrain()` çağrıları sırasında okuyucu endpoint'ler ile yarış koşulunu önlemek için | Yüksek |
| `RF n_jobs=1` (üretim) | Tek istek bazında CPU monopolizasyonunu engellemek için | Orta |
| Gradient Boosting | R² skorunu iyileştirmek için XGBoost/LightGBM alternatifleri | Orta |
| WebSocket Bildirimi | Scraper ilerlemesini gerçek zamanlı bildirmek için | Düşük |
| Periyodik Yeniden Dengeleme | Ardışık `insert()` çağrılarının ağaç dengesini bozmasını önlemek için belirli aralıklarla `build()` tetikleme | Düşük |

---

> **Not:** Bu rapordaki tüm satır numaraları ve fonksiyon isimleri, projenin `main` dalındaki son commit (`adf9ec6`) referans alınarak doğrulanmıştır.
