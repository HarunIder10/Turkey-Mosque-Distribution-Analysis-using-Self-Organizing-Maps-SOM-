import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import os

# --- Konfigürasyon ---
MOSQUE_DATA_FILE = 'mosques_by_province.csv'
GEOMETRY_FILE = 'tr.geojson'

# SOM Parametreleri
SOM_GRID_SIZE = 5  # 5x5 SOM grid
SOM_SIGMA = 1.5
SOM_LEARNING_RATE = 0.5
SOM_ITERATIONS = 1000

def normalize_turkish_text(text):
    """Türkçe karakterleri standartlaştırır"""
    replacements = {
        'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
        'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
        'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
    }
    for tr_char, en_char in replacements.items():
        text = text.replace(tr_char, en_char)
    return text.upper().strip()

def create_turkey_som_visualization():
    """SOM algoritması kullanarak cami dağılımını analiz eder ve görselleştirir"""
    try:
        print("🧠 SOM Analizi Başlatılıyor...\n")
        
        # 1. Veri Yükleme
        print("📊 Veriler yükleniyor...")
        df_mosque = pd.read_csv(MOSQUE_DATA_FILE)
        df_mosque.rename(columns={'Province': 'Province_Name'}, inplace=True)
        df_mosque['MosqueCount'] = pd.to_numeric(df_mosque['MosqueCount'], errors='coerce')
        
        gdf_turkey = gpd.read_file(GEOMETRY_FILE)
        
        # İl adı sütununu bul
        possible_columns = ['name', 'NAME_1', 'NAME', 'province']
        name_col = next((col for col in possible_columns if col in gdf_turkey.columns), None)
        
        if name_col is None:
            raise ValueError("İl adı sütunu bulunamadı!")
        
        gdf_turkey.rename(columns={name_col: 'Province_Name'}, inplace=True)
        
        # 2. İl adlarını standartlaştır
        gdf_turkey['Province_Name'] = gdf_turkey['Province_Name'].apply(normalize_turkish_text)
        df_mosque['Province_Name'] = df_mosque['Province_Name'].apply(normalize_turkish_text)
        
        # 3. Verileri birleştir
        gdf_merged = gdf_turkey.merge(df_mosque, on='Province_Name', how='left')
        # FIX: Pandas uyarısını düzelt
        gdf_merged = gdf_merged.fillna({'MosqueCount': 0})
        
        # 4. SOM için veri hazırlama
        print("🔧 SOM için veriler hazırlanıyor...")
        
        # Normalize edilecek özellikler
        features = gdf_merged[['MosqueCount']].values
        
        # Min-Max normalizasyon (0-1 arası)
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        
        # 5. SOM Eğitimi
        print(f"🎯 SOM eğitimi başlıyor ({SOM_GRID_SIZE}x{SOM_GRID_SIZE} grid, {SOM_ITERATIONS} iterasyon)...")
        
        som = MiniSom(
            x=SOM_GRID_SIZE,
            y=SOM_GRID_SIZE,
            input_len=1,
            sigma=SOM_SIGMA,
            learning_rate=SOM_LEARNING_RATE,
            random_seed=42
        )
        
        som.random_weights_init(features_normalized)
        som.train_random(features_normalized, SOM_ITERATIONS)
        
        print("✅ SOM eğitimi tamamlandı!")
        
        # 6. Her ili SOM grid'inde bir hücreye atama
        winner_coordinates = np.array([som.winner(x) for x in features_normalized])
        
        # SOM cluster'ı hesapla (0'dan başlayarak numaralandır)
        gdf_merged['SOM_Cluster'] = winner_coordinates[:, 0] * SOM_GRID_SIZE + winner_coordinates[:, 1]
        
        # Cluster'ları 1'den başlatmak için
        gdf_merged['SOM_Cluster'] = gdf_merged['SOM_Cluster'] + 1
        
        # 7. Görselleştirme
        fig = plt.figure(figsize=(20, 12))
        
        # Alt grafik 1: Orijinal Cami Sayıları
        ax1 = plt.subplot(1, 2, 1)
        gdf_merged.plot(
            column='MosqueCount',
            ax=ax1,
            legend=True,
            cmap='Reds',
            edgecolor='black',
            linewidth=0.5
        )
        # FIX: Legend'i manuel ekle
        cbar1 = ax1.get_figure().get_axes()[-1]
        cbar1.set_ylabel('Cami Sayısı', rotation=270, labelpad=20)
        
        ax1.set_title('Orijinal Cami Sayıları', fontsize=16, fontweight='bold', pad=15)
        ax1.set_axis_off()
        
        # Alt grafik 2: SOM Cluster'ları
        ax2 = plt.subplot(1, 2, 2)
        gdf_merged.plot(
            column='SOM_Cluster',
            ax=ax2,
            legend=True,
            cmap='tab20',
            edgecolor='black',
            linewidth=0.5,
            categorical=True
        )
        # FIX: Legend'i manuel ekle
        cbar2 = ax2.get_figure().get_axes()[-1]
        cbar2.set_ylabel('SOM Cluster', rotation=270, labelpad=20)
        
        ax2.set_title(f'SOM Kümeleme Sonuçları ({SOM_GRID_SIZE}x{SOM_GRID_SIZE})', 
                     fontsize=16, fontweight='bold', pad=15)
        ax2.set_axis_off()
        
        plt.suptitle('Türkiye İlleri Cami Dağılımı - SOM Analizi',
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Kaydet
        output_file = 'turkey_mosque_som.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n💾 Harita kaydedildi: {output_file}")
        
        plt.show()
        
        # 8. Cluster İstatistikleri
        print("\n📈 SOM Cluster İstatistikleri:")
        print("=" * 70)
        
        cluster_stats = gdf_merged.groupby('SOM_Cluster').agg({
            'MosqueCount': ['count', 'mean', 'min', 'max', 'sum'],
            'Province_Name': lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
        }).round(0)
        
        cluster_stats.columns = ['İl Sayısı', 'Ort. Cami', 'Min Cami', 'Max Cami', 'Toplam Cami', 'Örnek İller']
        print(cluster_stats)
        
        # 9. En çok camiye sahip iller
        print("\n🕌 En Çok Camiye Sahip 10 İl:")
        print("=" * 70)
        top_10 = gdf_merged.nlargest(10, 'MosqueCount')[['Province_Name', 'MosqueCount', 'SOM_Cluster']]
        top_10.columns = ['İl', 'Cami Sayısı', 'SOM Cluster']
        print(top_10.to_string(index=False))
        
        # 10. SOM U-Matrix (Distance Map)
        fig2, ax3 = plt.subplots(figsize=(10, 10))
        
        # U-Matrix hesapla
        distance_map = som.distance_map().T
        
        im = ax3.imshow(distance_map, cmap='bone_r', interpolation='nearest')
        ax3.set_title('SOM U-Matrix (Mesafe Haritası)', fontsize=16, fontweight='bold')
        ax3.set_xlabel('SOM Grid X')
        ax3.set_ylabel('SOM Grid Y')
        
        plt.colorbar(im, ax=ax3, label='Ortalama Mesafe')
        
        # Grid üzerine cluster numaralarını ekle
        for i in range(SOM_GRID_SIZE):
            for j in range(SOM_GRID_SIZE):
                cluster_num = i * SOM_GRID_SIZE + j + 1
                ax3.text(j, i, str(cluster_num), 
                        ha="center", va="center", color="red", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('turkey_mosque_som_umatrix.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 U-Matrix kaydedildi: turkey_mosque_som_umatrix.png")
        plt.show()
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()

# Fonksiyonu çalıştır
if __name__ == "__main__":
    # Not: SOM için 'minisom' kütüphanesi gereklidir
    # Kurulum: pip install minisom
    
    try:
        from minisom import MiniSom
        create_turkey_som_visualization()
    except ImportError:
        print("⚠️ 'minisom' kütüphanesi bulunamadı!")
        print("Lütfen şu komutu çalıştırın: pip install minisom")
        print("\nAlternatif olarak temel görselleştirme için 'turkey_mosque_basic' kodunu kullanın.")