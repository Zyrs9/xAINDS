# AI-Driven Lightweight NIDS: An eBPF-Powered Edge-to-Cloud Architecture 

Bu proje, IEEE standartlarında akademik bir Ağ Saldırı Tespit Sistemi (NIDS) prototipidir. Sistem, yüksek performanslı **eBPF/XDP** (Data Plane) teknolojisini, **Cost-Sensitive (Maliyet Duyarlı) Machine Learning** algoritmalarını ve **Counterfactual SHAP** tabanlı Açıklanabilir Yapay Zeka'yı (XAI) bir araya getirerek sıfır-RTT (%100 tel hızında) çalışan, şeffaf ve güçlü bir güvenlik mimarisi sunar.

---

## 1. Klasör ve Mimari Yapısı

Proje, kavramların (Data Plane, ML, XAI, API) birbirinden kesin sınırlarla ayrıldığı modüler bir mimariye sahiptir.

| Dizin | Amaç ve Açıklama |
|---|---|
| `nvp_baseline/` | **Kum Torbası (Legacy):** Projenin eBPF'e geçmeden önceki `Scapy` tabanlı, kullanıcı alanı (user-space) darboğazına sahip eski ve yavaş versiyonu. Sadece performans benchmark testlerinde "eBPF'in neden gerekli olduğunu" kanıtlamak için dondurulmuş bir referans noktası olarak tutulmaktadır. |
| `src/ebpf/` | **Zero-Copy Data Plane:** Sistem ağ trafiğini dinleyen ve özellik çıkartan (feature extraction) çekirdek (kernel) modülleridir. |
| `src/ml/` | **Maliyet Duyarlı Çıkarım Motoru:** Gerçek ağ verisi (UNSW-NB15) ile eğitilen ve SOC (Security Operations Center) uyarı yorgunluğunu önleyen ML motoru. |
| `src/xai/` | **Açıklanabilir Yapay Zeka (XAI):** "Yapay Zeka bu kararı neden verdi?" sorusunu Neden-Sonuç (Causality) prensibiyle ispatlayan açıklayıcı modül. |
| `src/api/` | **Dışa Aktarım (Presentation):** Kararların dış sistemlere veya bir UI Dashboard'a iletilmesini sağlayan uç nokta. |
| `infrastructure/` | **DevOps:** Sistemin farklı ortamlarda sorunsuz çalıştırılabilmesi için gerekli olan konteynerizasyon dosyaları. |
| `benchmark/` | **Akademik Kanıt:** eBPF mimarisi ile `nvp_baseline` arasındaki CPU/RAM farklarını ispatlamak için yazılan canlı test script'leri. |

---

## 2. Dosyaların Etkileşim Haritası ve Görevleri

Sistemdeki kritik dosyaların birbirleriyle nasıl bir yaşam döngüsü içerisinde çalıştığı aşağıda açıklanmıştır:

### A) Ağ Veri Düzlemi (Data Plane)
*   **`src/ebpf/flow_tracker.c`**: Ağ kartına (NIC) tutunan bir XDP C programıdır. Paketler henüz işletim sistemine (Linux Kernel) girmeden onları yakalar, UNSW-NB15 veri setine uygun 10 istatistiksel özelliği saniyede milyonlarca paket hızında (Tel hızı - Wire speed) hesaplar ve bir Ring Buffer üzerinden User-Space'e iletir.
*   **`src/ebpf/user_space.py`**: C programından gelen verileri dinleyen Python Köprüsü (Bridge). BCC (BPF Compiler Collection) kütüphanesini kullanarak Kernel'deki C struct verilerini alır, Numpy dizisine çevirir ve anlık olarak `engine.py` dosyasına gönderir.

### B) Makine Öğrenmesi (Machine Learning Core)
*   **`src/ml/train_baseline.py`**: Geliştiricinin bir kez çalıştırdığı eğitim betiği. `NF-UNSW-NB15-v2.csv` veri setini okur, Random Forest ile en önemli 10 özelliği seçer, İzolasyon Ormanı (Isolation Forest) modelini eğitir. Siber güvenlikte False Negative (atağı kaçırma) cezası False Positive'den yüksek olduğu için (1'e 10 kuralı) Maliyet Duyarlı Threshold (Eşik) hesaplar ve `.pkl` dosyalarını kök dizine kaydeder.
*   **`src/ml/engine.py`**: Gerçek zamanlı (Live-inference) tahmin motoru. `user_space.py` veya `server.py` tarafından beslenir. Ayrıca üst üste gelen paketlerin uyarı yorgunluğunu engellemek için (Alert Fatigue) `SpikeDetector` mekanizması barındırır.

### C) Açıklanabilirlik (Explainable AI - XAI)
*   **`src/xai/explainer.py`**: Makine öğrenmesinden çıkan kararın arkasındaki nedeni arar. SHAP (SHapley Additive exPlanations) kullanılarak hangi özelliğin anomaliteye ne kadar etki ettiği hesaplanır. Doğal Dil Üretimi (NLG) motoru ve **MITRE ATT&CK** veritabanı eşleştirmesi ile SHAP değerlerini *(örn: T1048 - Exfiltration)* SOC analisti için insan diline çevirir.
*   **`src/xai/counterfactual_demo.py`**: Projenin Masterpiece (Sanat Eseri) dosyasıdır. Jürinin "Bu model gerçekten olayları anlıyor mu, yoksa ezberledi mi?" sorusunu yanıtlar. Modelin anomalite kararı kıldığı bir ağ trafiğindeki en baskın özelliği (Örn: OUT_BYTES) bulur. Kullanıcıdan bu sayıyı yapay olarak %50 düşürmesini ister. Yeni simülasyonda kararın ANOMALY'den NORMAL'e döndüğünü matematiksel olarak ispatlar (Causality Proven).

### D) Altyapı ve Sunum
*   **`src/api/server.py`**: Tüm sistemin bulut sekansı. Dış dünyadan JSON formatında trafik kabul edebilen, model kararlarını dışarı açan ve `/explain` rotası ile MITRE raporlarını dışa aktaran modern bir FastAPI uygulaması.
*   **`benchmark/metrics_collector.py` & `traffic_replay.sh`**: Canlı bir deneme ortamında (PCAP dosyasını tekrar oynatarak) `nvp_baseline` ile yeni eBPF mimarisinin CPU/RAM tüketimlerini milisaniyelik boyutta toplayıp, makale grafikleri için `.csv` formatında dışarı veren sistem.
*   **`Makefile`**: Bütün sistemi komut satırı tek satırlık komutlarla (örn: `make train`, `make deploy`) yönetmeyi sağlayan orkestratör betik.

---

## 3. Çevre Kurulumu ve Çalıştırma Adımları

Sistem eBPF kullandığı için gereksinimleri platformlara göre değişiklik gösterir.

### Seçenek 1: Canlı Ağ İzleme Modu (Linux / Ubuntu 22.04+)
> Bu sürüm sistemin tüm özellikleriyle (Gerçek zamanlı XDP paket okuma) çalıştığı üretim (Production) ve test ortamıdır.

**Gereksinimler:**
*   **Linux Kernel 5.10+** (Modern BPF yardımcı programları ve CO-RE/BCC stabilitesi için zorunludur).
*   Donanım Sanallaştırması veya Fiziksel Makine (`eth0`, `ens33` gibi gerçek bir NIC Interface'i).
*   Gerekli kütüphaneler: `sudo apt install bpfcc-tools python3-bpfcc libbpf-dev clang llvm linux-headers-$(uname -r)`

**Çalıştırma Adımları:**
1. Veri seti kök dizinde (`NF-UNSW-NB15-v2.csv`) bulunmalıdır.
2. Bağımlılıkları kurun: `make install`
3. Arka plan modeli eğitin: `make train` (Bu adım `.pkl` model dosyalarını üretecektir).
4. Data Plane'i dinlemeye ve gerçek zamanlı tahminleri görmeye başlayın: `sudo python -m src.ebpf.user_space --interface eth0`

### Seçenek 2: Geliştirme, Analiz ve Sunum Modu (Windows/macOS)
> Makineniz Kernel 5.10 desteklemiyorsa, işletim sisteminden bağımsız olarak model eğitme, Counterfactual XAI çalışması yapma ve API server'ı ayağa kaldırma imkanı sunar. XDP veri çekme özelliği offline simüle edilir.

**Gereksinimler:**
*   Python 3.10+

**Çalıştırma Adımları:**
1. Bağımlılıkları kurun: `pip install -r requirements.txt`
2. Modeli eğitin: `python -m src.ml.train_baseline`
3. **Akademik Demoyu İzleyin:** Nedensellik (Causality) analizinin nasıl çalıştığını ispatlayan konsolu başlatın: `python -m src.xai.counterfactual_demo --auto`
4. **API Sunucusunu Başlatın:** `python -m src.api.server` komutu ile port 8000 üzerinden analiz API'sini açın. Dilerseniz offline modda simüle ağ trafik verisi göndermek için `python -m src.ebpf.user_space --offline` kullanabilirsiniz.

### Seçenek 3: Docker-Compose (Uçtan Uca Deployment)
> İzole bir ortamda eBPF ve API sunucusunu birlikte kaldırmak için kullanılır (Host makine Linux olmalıdır).

1. `make build` (İmajları derler)
2. `make deploy` (Root yetkileri ile XDP listener ve API Backend'i ayağa kaldırır)

---

## 4. Gelecek Adımlar (ROADMAP / TO-DO)

Bu modüller şu anda başarıyla çalışsa da projenin "Ürün" (Product) seviyesine gelmesi ve makalenin gücünü katlaması için şu adımlar planlanmaktadır:

- [ ] **Görsel Ön Yüz (Frontend Dashboard):**
  - **Açıklama:** Terminal çıktılarını ve API üzerindeki verileri anlık grafiklere (Chart.js / Recharts) dökecek olan şık bir Next.js veya React web arayüzü yazılması.
  - **Amacı:** MITRE ATT&CK matris eşleşmelerinin ve SHAP değerlerinin jüri/kullanıcıya görsel şov eşliğinde gösterilmesi.
- [ ] **Kalıcı Loglama (PostgreSQL / SQLite Storage):**
  - **Açıklama:** Belirlenen uyarıların, maliyet kaybı istatistiklerinin salt API belleğinden çıkarılıp ACID prensipleriyle kalıcı bir veritabanlarına işlenmesi.
  - **Amacı:** Geriye dönük sorgulama (Querying) yapılabilmesi ve bir siber olay tepki (Incident Response) modülü yazılması için altyapı hazırlanması.
- [ ] **Grafik Üreteçleri (Matplotlib Integration):**
  - **Açıklama:** `benchmark/metrics_collector.py` ve SHAP değerlerinden çıkan matematiksel hesaplamaları doğrudan PDF üzerinde basılabilecek LaTeX veya PGFPlots grafikleri şeklinde dışa aktaran script'in modüle bağlanması.
  - **Amacı:** Yazılacak IEEE Paper'ın Experiment & Results bölümünün direkt model verisinden, hiç el değmeden üretilmesi.
- [ ] **Kapsamlı Sızma Testleri:**
  - **Açıklama:** VMWare üzerindeki izole test topolojisinde (Kali Linux & Ubuntu Node), gerçek metotlar *(örn: Hping3, Nmap, Metasploit)* kullanarak eBPF çekirdeğinin yakaladığı trafikler ile Isolation Forest'ın eş zamanlı reaksiyon vermesinin videoya alınması.
