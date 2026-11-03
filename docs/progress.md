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
- Sonraki iş: Colab'da A100 seç, Drive bağla, `python -m src.data`, smoke test, 1 epoch hız ölçümü, runtime'ı kapat (F4 başlangıcı).

## Sonraki adım
- Kullanıcı makaleleri okuyup taslak notları düzeltir, arkadaşına gönderir.
- Arkadaşından `okuma_notlari.md` dosyası istenecek.
- Sonra 4. adım: öğrenme artifact'ı.
