## Curriculum ve bandit tarafı

### Graves ve ark. 2017 — Automated Curriculum Learning for Neural Networks (ICML)
- **İddia:** Hangi görevden örnek verileceğini non-stationary bir bandit (Exp3.S) seçebilir. Ödül learning progress'tir. Uygun sinyalle öğrenme bazı durumlarda yaklaşık iki kat hızlanır.
- **Nasıl test etmiş:** LSTM; n-gram, repeat copy ve bAbI görevleri; 10 seed; uniform örneklemeye karşı. Maximum likelihood'da en tutarlı sinyal Prediction Gain (PG), variational'da GVCG. Uniform çok güçlü bir baseline çıkmış.
- **Bizim soruya ne diyor:** Yöntemimizin iskeleti bu. Kol = (blur, contrast) seviyesi, ödül = PG, yani aynı batch'te update öncesi ve sonrası loss farkı (tek ek forward pass). PG, gradient varyansı yüksek kollara kayar (Sec. 3.3). Bu yüzden "önce bulanık" sonucunu SPG ve uniform kolla kontrol etmeliyiz.

### Wang ve ark. 2023, 2024 — EfficientTrain (ICCV) ve EfficientTrain++ (TPAMI)
- **İddia:** Her epoch tüm veriyi kullan, ama önce kolay örüntüleri göster: düşük frekans (Fourier cropping) ve zayıf augmentation. ImageNet eğitimi accuracy kaybı olmadan 1.5x (ET) ve 1.5–3x (ET++) hızlanır.
- **Nasıl test etmiş:** ImageNet-1K/22K; ResNet, ConvNeXt, DeiT, Swin, CSWin ve MAE. Schedule'ı bir kez offline arama ile Swin-T üzerinde bulmuş. Sonra tüm modellerde sabit kullanmış.
- **Bizim soruya ne diyor:** En yakın prior work. Farkımız dört noktada: sırayı biz dayatmıyoruz, bandit online seçiyor; ödül learning progress, accuracy ya da compute değil; eksen blur ve contrast; robustness ve shape bias ölçüyoruz. ET ikisini de ölçmüyor, contrast ve bebek görmesinden hiç söz etmiyor. Onların sabit merdiveni bizim için hazır bir baseline kolu.

### Wu, Dyer, Neyshabur 2021 — When Do Curricula Work? (ICLR)
- **İddia:** Standart eğitimde curriculum, anti-curriculum ve rastgele sıra aynı sonucu verir. Küçük kazanç sıradan değil, veri setinin zamanla büyümesinden (pacing) gelir. Curriculum sadece kısa bütçede ve noisy label'da işe yarar.
- **Nasıl test etmiş:** CIFAR-10/100 ve FOOD-101 üzerinde ResNet-50; 3 sıra × 180 pacing fonksiyonu × 3 seed, toplam 25.000'den fazla model. Kısa bütçe (yaklaşık 5 epoch ve altı) ve %20–80 label noise ayrıca denenmiş.
- **Bizim soruya ne diyor:** Hakemin sorusu şu olacak: "Kazanç sıradan mı geliyor, blur görmüş olmaktan mı?" Bu yüzden şu kollar şart: zamanda karıştırılmış bandit schedule'ı, sharp-first ve sabit rastgele blur/contrast augmentation. En az 3 seed, eşit compute, kısa ve tam bütçe. Makale online curriculum'ı ve çözünürlük curriculum'ını kapsam dışı bırakıyor (Sec. 3), yani bizim tasarımı test etmemiş.

### Hacohen & Weinshall 2019 — On The Power of Curriculum Learning in Training Deep Networks (ICML)
- **İddia:** Curriculum iki parçadır: scoring function (örnek ne kadar zor) ve pacing function (zor örnekler ne hızla eklenir). Kolaydan zora örneklemek eğitimi hızlandırır ve son accuracy'yi biraz artırır.
- **Nasıl test etmiş:** CIFAR-10/100 ve küçük bir ImageNet alt kümesi; küçük CNN ve VGG. Zorluk skoru önceden eğitilmiş Inception + SVM'den ya da aynı ağın önceki eğitiminden geliyor. Rastgele skor vanilla ile aynı. O anki ağın loss'uyla skorlamak (self-paced) zarar veriyor. Kazanç küçük, şekillerden okunduğu kadarıyla 1–2 puan.
- **Bizim soruya ne diyor:** Bizde ayrı f ve g yok. Bandit'in tek kol seçimi hem sırayı hem süreyi belirliyor. İkisini koşudan sonra log'dan ayırırız: sıra = ziyaret edilen kolların bulanıktan keskine sırayla korelasyonu, süre = her kolda kalınan adım sayısı. Pacing etkin learning rate'i değiştirir, bu yüzden LR schedule tüm kollarda aynı olmalı.

### Portelas ve ark. 2019 — Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments (CoRL)
- **İddia:** Absolute learning progress (ALP) ile çalışan bir teacher (ALP-GMM), sürekli parametreli ortamlarda ön bilgi olmadan RL ajanını yönlendirebilir. Gereksiz boyutlara ve öğrenilemez bölgelere dayanıklıdır.
- **Nasıl test etmiş:** SAC ajanı, BipedalWalker (2D Stump Tracks, 12D Hexagon Tracks), 32 seed; Random, RIAC, Covar-GMM ve elle yapılmış Oracle'a karşı. ALP-GMM çoğu durumda en iyi (Hexagon'da %80'e karşı Oracle %68). Kısa bacaklı ajanda Oracle kazanıyor.
- **Bizim soruya ne diyor:** (blur, contrast)'ı sürekli 2D uzay alırsak teacher bu. ALP = |yeni ödül − en yakın eski noktanın ödülü|. Mutlak değer unutmayı yakalar: ağ keskin görüntüde gerilerse teacher oraya döner. Varsayılanımız Exp3.S kalır. ALP-GMM'i "sonuç grid'in artefaktı değil" kontrolü olarak koşarız.

---

## Hocanın bildirisi (ortak okuma, benim notum)

### Arslan, Jang, Sinha 2024 — Internal Neural Noise Progression for Emergent Classification Robustness (CCN)
- **İddia:** Ağın iç gürültüsünü eğitimde yüksekten düşüğe indirmek, bozulmuş girdide sınıflandırmayı iyileştirir. Etki stochastic resonance'a benziyor.
- **Nasıl test etmiş:** AlexNet. Deney 1: CIFAR-10 ve Fashion-MNIST; çözünürlük (Gaussian blur) ve sondan bir önceki katmandaki gürültü için LH, HL ve sabit kombinasyonları; 15 + 15 epoch iki faz. Deney 2: ImageNet; her katmana batch varyansının %25'i kadar gürültü; clear, noise, noise-to-clear, clear-to-noise. Test: Gaussian ve salt-and-pepper gürültülü girdi. Eğitimde bozuk görüntü hiç yok.
- **Bizim soruya ne diyor:** Önce düşük çözünürlük robustness veriyor, yüksekten düşüğe gürültü bunu artırıyor (Deney 1). Deney 2'de fark küçük ("not substantial"). Schedule'lar elle ve sabit. Bizim soru bunun bir adım ötesi: ağ aynı sırayı kendisi seçer mi? Contrast bildiride yok, bizim ekimiz.
