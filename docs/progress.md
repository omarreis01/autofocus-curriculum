# CMPE492 — İlerleme

## Proje
- Ad: **Auto-Curriculum**
- Ekip: Ömer Faruk Çelik (2021400084), Mustafa Taha Söylemez (2021400081)

## Araştırma sorusu
Ağa blur ve contrast seviyesini kendisi seçtirsek, bebek gibi önce bulanık ve soluk görüntüden başlar mı? Bu ona robustness kazandırır mı?
- Seviyeyi bir bandit seçer. Reward sadece learning progress (Graves 2017).
- Robustness ve shape bias reward'a girmez. Sadece sonda ölçülür.
- Danışman: Suayb S. Arslan (CCN 2024 bildirisi: Arslan, Jang, Sinha).

## Yol haritası (6 adım)
1. Tek cümle — bitti
2. Makale listesi — bitti
3. Okuma — **şu an burada** (2026-10-11)
4. Öğrenme artifact'ı
5. Plan ve deneyler
6. 1 sayfalık özet

## Görev dağılımı (3. adım)
- Ben: curriculum ve bandit tarafı — Graves 2017, EfficientTrain/++, Wu 2021, Hacohen 2019, Portelas 2019 (ALP-GMM).
- İkimiz: hocanın CCN 2024 bildirisi.
- Teslim: `okuma_notlari.md` içindeki "Curriculum ve bandit tarafı" bölümü (iddia / nasıl test etmiş / bizim soruya ne diyor). Dosya henüz bu klasörde yok.

## Durum (2026-10-11)
- Beş makale ve CCN bildirisi okundu. Ajan notları: `docs/okuma/*.md`.
- Gönderilecek bölüm taslağı: `docs/okuma/okuma_notlari_curriculum.md`.
- Okuma rehberi (HTML): https://claude.ai/artifact/TV8391ULJSGfaBttn5xSrd
- Okumadan çıkan ana kararlar (taslak): Exp3.S + PG ile başla, SPG kontrol. Kontrol kolları: uniform, karıştırılmış schedule, sharp-first, sabit blur/contrast augmentation, elle merdiven. En az 3 seed, kısa + tam bütçe. ALP-GMM sadece kontrol.
- 2026-10-13: Rehber tam anlatıma genişletildi (v2). Yeni bulgular: Jang & Tong 2021 (nesnelerde blur→clear robustness kalmıyor), Jang & Tong 2024 (karışık blur güçlü baseline), Katzhendler & Weinshall 2019 (maruz kalma itirazı). CIFAR-10-C'de blur ve contrast eğitimle örtüşüyor, görülen/görülmeyen ayrı raporlanmalı. Shape bias seti CIFAR'a uymuyor, açık soru.

- 2026-10-13: 1 sayfalık proje taslağı (İngilizce, Claude Docs): https://claude.ai/code/artifact/37bee1d6-2ccb-49c0-b6b1-2048aa784f28 . İçerik: motivasyon, H1/H2, yöntem, 7 koşul, fark, aşamalı to-do, hocaya sorular. Varsayım: CIFAR-10 + ResNet-18 (onaylanacak). 2026-10-13: Proposal sürümüne çevrildi (daha az teknik, tarihsiz iş planı). Hocaya gidecek PDF: `docs/project_proposal.pdf` (11pt, 2 sayfa). Docs dokümanı aynı metinle eşitlendi.

- 2026-11-03: Colab MCP kuruldu (`googlecolab/colab-mcp`, local scope, `uvx`). Sadece session-proxy modu var: araç varsayılan tarayıcıda Colab sekmesi açar, sekme açık kalmalı, Google girişi kullanıcıda. Araçlar yeni oturumda görünür. Plan: önce kısa bütçeli pilot, sonra tam deneyler; checkpoint'ler Drive'a.

- 2026-11-03: Colab Pro, 300 CU. Deney planı: `docs/plans/colab-deneyler.md` (F0–F8, tahmini ~225 CU). Varsayılanlar: 16 kol, Exp3.S+PG, 30/100 epoch, CIFAR-10-C. Shape bias ertelendi.
- 2026-11-03: Kod repo'su: https://github.com/omarreis01/autofocus-curriculum (public, hoca da görür). Colab token'sız `git clone` ile çeker. Commit'ler kullanıcı adına, Claude satırı yok.
- 2026-11-03: F0–F1 bitti (repo'da, 8 test geçiyor, GPU'da henüz koşmadı). Kurulum: `colab/README.md`. Colab MCP artık Chrome'da açılır (`~/.claude.json` içinde `BROWSER=open -a "Google Chrome" %s`); Chrome eklentisi GPU menüsünü sürer, Colab sekmesi Claude sekme grubunda olmalı. Drive izni kullanıcının tıklaması ister.
- 2026-11-03: İlk GPU koşusu (A100). Veri HF mirror'dan iner (CIFAR-10: `liangnanying/cifar-10-python`, md5 kontrollü; CIFAR-10-C: `robro/cifar10-c-parquet`, etiket sırası CIFAR-10 test ile birebir doğrulandı). Toplam ~1.5 dk; Zenodo/Toronto çok yavaştı.
- Ölçüm: 2.0 s/epoch (eager), `--compile` 1.7 s ama 30 s derleme → kullanılmıyor. Tam CIFAR-10-C eval 15 s. Sharp 30 epoch: clean %93.2, seen %70.5, unseen %73.7 → F1 kabul geçti. Sonuç Drive `results/sharp/0/`.
- Colab kontrolü: Colab MCP hücre ekler/çalıştırır ama durduramaz/restart edemez. Bunlar için Colab sekmesi bu session'ın Chrome tab grubunda olmalı (her yeni session yeni grup açar).
- 2026-11-03: F2+F3 bitti (commit a689f01, 24 test). Exp3.S η=0.01 (Graves 1e-3 bizim 3k–10k adımda kıpırdamıyor), ε=0.05, α=1/t. 7 koşul × 3 seed × {short, full} config'i `configs/`. Kararlar: `docs/plans/colab-deneyler.md` "F2/F3 kararları".
- 2026-11-03: Pilot başlatıldı (short, seed 0, 7 koşul; 6'sı aynı A100'de paralel, sonra shuffled). Sonuç/log: Drive `results/short/<koşul>/0/`, `logs/short_<koşul>_0.log`. Kullanıcı PC'yi kapattı; koşu yarıda kaldıysa aynı hücre `--resume` ile devam eder.
- 2026-11-03: Pilot bitti (7/7). Rapor: https://claude.ai/artifact/4uPcbPyQs3RmhzjeWz8FAi . Bulgu 1: BN uyumsuzluğu — batch başına tek kol, eval'de karışık running stats; random/teacher/shuffled/reverse clean %13–60. Temiz train verisiyle BN recal sonrası normal (random 89.1, teacher_spg 87.2). Bulgu 2: teacher bulanık→keskin seçmiyor; uniformdan en bozuk kola (σ3/c0.2) gidiyor (H1'in tersi, 1 seed). Bulgu 3: recal sonrası random ≈ teacher ≈ shuffled (C ~%79–82), sharp 69/73.
- 2026-11-04: BN kararı A (kullanıcı "sorma" dedi): eval öncesi temiz train ile BN recal (`src/eval.py`, geri alma `--no-recal`). F5 bitti (7×3×30 ep, std ≤0.8). Clean/seen/unseen: sharp 93.2/69.6/73.3, staircase 90.7/75.8/78.7, reverse 84.4/78.1/71.3, random 89.4/81.4/82.2, shuffled 84.2/79.7/78.5, teacher_pg 84.8/80.1/78.8, teacher_spg 87.0/81.0/80.3. Teacher 3 seed'de de en bozuk kola gidiyor (hard-first, H1 değil). Random en iyi. Blur eğitimi gürültüye robustness veriyor (55→76); contrast eğitimi contrast'a yardım etmiyor (BN batch stats kontrastı siliyor). Rapor (v2): https://claude.ai/artifact/4uPcbPyQs3RmhzjeWz8FAi . Özet JSON: Drive `results/short_summary.json`.
- 2026-11-04: F6 başladı (full, 100 ep): `colab/run_budget.sh full 0 1` sonra `full 2`, ~2 saat. Log: Drive `logs/f6_runner.log`.
- Sonraki iş: F6 sonuçları + açık sorular: PG ödülü zor kola mı kayıyor (PG/loss_before teacher'ı), contrast'ı gerçek yapmak, shape bias.

## Sonraki adım
- Kullanıcı makaleleri okuyup taslak notları düzeltir, arkadaşına gönderir.
- Arkadaşından `okuma_notlari.md` dosyası istenecek.
- Sonra 4. adım: öğrenme artifact'ı.
