# Colab deney planı (2026-11-03)

Amaç: proposal'daki deneyleri (Bölüm 4) Colab Pro'da 300 işlem birimi (CU) ile koşmak.
Kaynak: `docs/project_proposal.pdf`, `docs/okuma/okuma_notlari_curriculum.md`.

## Kurallar
- Kod notebook hücresinde yaşamaz. Kod bir repoda durur (`src/`), Colab onu çeker ve çalıştırır. Notebook sadece başlatıcıdır.
- Uzun koşular Colab'da arka planda script olarak çalışır (`nohup python -m ... &`). Log ve checkpoint Drive'a yazılır. Sekme kapanırsa koşu checkpoint'ten devam eder.
- Her koşu tek bir config dosyasıyla tanımlanır (koşul, seed, bütçe). Sonuç `results/<koşul>/<seed>/` altına JSON olarak yazılır.
- Boşta GPU runtime açık kalmaz. Her faz sonunda runtime kapatılır.

## Varsayılan kararlar (proposal'daki açık sorular; geri almak kolay)
| Karar | Varsayılan | Nasıl değişir |
|---|---|---|
| Veri, model | CIFAR-10, ResNet-18 (CIFAR varyantı) | config |
| Kollar | 4 blur (σ = 0, 0.75, 1.5, 3 px) × 4 contrast (1.0, 0.6, 0.35, 0.2) = 16 kol | config |
| Teacher | Exp3.S + PG ödülü, Graves 2017 quantile ölçekleme | config |
| Alternatif sinyal | SPG (aynı koldan yeni batch) | config |
| Robustness | CIFAR-10-C; "görülen" = gaussian/defocus/glass/motion/zoom blur + contrast, "görülmeyen" = geri kalan 13 | eval script |
| Shape bias | Ertelendi. Faz 4'te karar | — |
| Bütçe | kısa = 30 epoch, tam = 100 epoch; aynı LR schedule tüm koşullarda | config |
| Blur/contrast | GPU'da batch üzerinde uygulanır (CPU dataloader yok) | — |

## Görevler

### F0 — Kurulum (CPU, ~0 CU)
- T0.1 Repo: `src/` iskeleti, `pyproject`/`requirements.txt`, git. Colab `git clone` ile çeker.
- T0.2 Colab başlatıcı notebook: Drive mount, repo clone/pull, GPU kontrol, CIFAR-10 ve CIFAR-10-C'yi Drive'a bir kez indir.
- Kabul: Colab'da tek hücre `python -m src.smoke` çalışır, Drive'da veri durur.

### F1 — Pipeline (yerelde CPU + Colab'da kısa GPU testi, ~5 CU)
- T1.1 `degrade.py`: batch tensöründe Gaussian blur (σ) ve contrast (ortalamaya doğru ölçekleme). Test: σ=0 ve c=1 görüntüyü değiştirmez; contrast ortalamayı korur; blur yüksek frekans enerjisini azaltır.
- T1.2 `train.py`: GPU'da tutulan CIFAR-10, ResNet-18, AMP, sabit LR schedule. Her adımda bir `Schedule` nesnesinden (blur, contrast) seviyesi alır. Her adım için seçilen kolu loglar.
- T1.3 `eval.py`: clean accuracy, CIFAR-10-C (19 bozulma × 5 şiddet), görülen/görülmeyen ayrı ortalama.
- Kabul: "sadece keskin" koşulu 30 epoch'ta ~%93 clean accuracy verir (ResNet-18 için bilinen değer). Eval tek modelde 5 dakikadan kısa sürer.

### F2 — Teacher (CPU'da test, GPU'da kısa koşu, ~10 CU)
- T2.1 `teacher.py`: Exp3.S (Graves 2017, Alg. 1), PG ve SPG ödülleri, quantile ölçekleme.
- T2.2 Sağlık testi: öğrenmeyen ağ (LR = 0) ile teacher sıra üretmez, kol dağılımı uniforma yakın kalır.
- T2.3 Sentetik test: bir kolun ödülü sabit yüksekse teacher o kola yoğunlaşır.
- Kabul: iki test geçer; gerçek koşuda kol seçimi zamanla değişir ve loglanır.

### F3 — Kontrol koşulları (~0 CU, kod)
- T3.1 Sadece keskin; batch başına rastgele seviye; elle bulanık→keskin merdiven; keskin→bulanık.
- T3.2 Karıştırılmış bandit schedule: bir bandit koşusunun kol logunu zamanda karıştırıp tekrar oynatır. Kol sayıları birebir aynı kalır.
- Kabul: her koşul aynı adım sayısını ve aynı LR schedule'ı kullanır (testle).

### F4 — GPU kıyası ve pilot (~10 CU)
- T4.1 T4, L4 ve A100'de 1 epoch süresini ölç. Kaynaklar panelinden saatlik CU'yu oku. "CU / koşu" en düşük GPU'yu seç.
- T4.2 Pilot: 7 koşul × 1 seed × kısa bütçe. Hata ve log kontrolü.
- Kabul: bütçe tablosu gerçek ölçümle güncellenir.

### F5 — Kısa bütçe (~30 CU)
- 7 koşul × 3 seed × 30 epoch = 21 koşu + CIFAR-10-C eval.
- Ara rapor: H1 (bandit bulanık/soluktan mı başlıyor), ilk robustness farkları.

### F6 — Tam bütçe (~90 CU)
- 7 koşul × 3 seed × 100 epoch = 21 koşu + eval.

### F7 — Sağlamlık kontrolleri (~30 CU)
- ALP-GMM teacher (sürekli 2D uzay), 3 seed × tam bütçe.
- Shape bias ölçümü (karar sonrası).

### F8 — Analiz ve rapor (CPU)
- Kol seçim ısı haritası (zaman × kol), sıra korelasyonu, koşullar arası robustness, seed dağılımıyla.

## Bütçe tahmini
**Ölçüm (2026-11-03, A100):** 2.0 s/epoch, 30 epoch koşu ~80 s, 100 epoch ~3.5 dk, eval 15 s. F5 (21 kısa koşu) ~35 dk, F6 (21 tam koşu) ~80 dk; PG ek forward'u ile ~1.3×. Aşağıdaki tablo eski tahmindir, gerçek ihtiyaç çok daha düşük.

Tahmin, ölçüm değil. F4'te gerçek sayılarla güncellenecek.
Varsayım: A100 ~12 CU/saat, ResNet-18 GPU-veri ile ~6 s/epoch, bandit koşularında PG için ek forward (~1.3×).

| Faz | Tahmini CU |
|---|---|
| F1–F2 geliştirme | 15 |
| F4 kıyas + pilot | 10 |
| F5 kısa bütçe | 30 |
| F6 tam bütçe | 90 |
| F7 kontroller | 30 |
| Yeniden koşu payı | 50 |
| **Toplam** | **~225 / 300** |

Risk: Colab Pro A100'ü garanti etmez. L4 daha ucuz ama yavaş olabilir; F4 hangisinin daha az CU harcadığını gösterir.
