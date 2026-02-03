import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

VERI_YOLU = "Veri_Seti_Gorsel"
KAYIT_ADI = "Fark_Analizi.png"

def karsilastirma_gorseli_olustur():
    real_path = os.path.join(VERI_YOLU, "REAL")
    fake_path = os.path.join(VERI_YOLU, "FAKE")
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print("HATA: Veri seti klasörleri bulunamadı! 'Veri_Seti_Gorsel' klasörünü kontrol et.")
        return
    
    real_img_name = random.choice(os.listdir(real_path))
    fake_img_name = random.choice(os.listdir(fake_path))
    
    img_real = mpimg.imread(os.path.join(real_path, real_img_name))
    img_fake = mpimg.imread(os.path.join(fake_path, fake_img_name))


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(img_real)
    ax[0].set_title("GERÇEK SES (Real)\n(Doğal Frekans Dağılımı)", color="green", fontsize=14, fontweight='bold')
    ax[0].axis("off")
    ax[0].text(10, 110, "Homojen ve\nDoğal Geçişler", color="white", fontsize=10, 
               bbox=dict(facecolor='green', alpha=0.5))

    ax[1].imshow(img_fake)
    ax[1].set_title("SAHTE SES (Deepfake)\n(Yapay Zeka İzleri)", color="red", fontsize=14, fontweight='bold')
    ax[1].axis("off")
    ax[1].text(10, 110, "Keskin Çizgiler &\nYapay Boşluklar", color="white", fontsize=10, 
               bbox=dict(facecolor='red', alpha=0.5))

    plt.suptitle("İnsan Kulağının Duyamadığı Dijital Parmak İzleri", fontsize=16)
    plt.tight_layout()
    plt.savefig(KAYIT_ADI, dpi=300)
    print(f"Görsel oluşturuldu: {KAYIT_ADI}")
    plt.show()

if __name__ == "__main__":
    karsilastirma_gorseli_olustur()