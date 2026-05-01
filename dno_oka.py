'''Analiza dna oka - segmentacja naczyń krwionośnych'''
import numpy as np
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import tifffile

# ===== KONFIGURACJA =====
CONFIG = {
    'sciezka_img': 'dno_oka/dataset/images/',
    'sciezka_mask': 'dno_oka/dataset/mask/',
    'margin': 10,
    'frangi_sigmas': range(1, 3),
    'frangi_scale_range': (1, 6),
    'clahe_clip': 0.03,
    'adapthist_clip': 0.1,
}

# Generowanie ścieżek
numery = ['01']
choroby = ['h']
obrazy = [f"{CONFIG['sciezka_img']}{num}_{ch}.JPG" for num in numery for ch in choroby]
maski = [f"dno_oka/dataset/manual1/{num}_{ch}.tif" for num in numery for ch in choroby]  # Expert annotations

# ===== WCZYTYWANIE =====
def wczytaj_obraz(sciezka):
    '''Wczytuje obraz JPG i konwertuje do skali szarości'''
    with open(sciezka, 'rb') as f:
        return np.array(Image.open(f).convert('L'))

def wczytaj_maske(sciezka):
    '''Wczytuje expert annotation z manual1 - białe piksele (255) to naczynia'''
    maska = tifffile.imread(sciezka)
    if maska.ndim == 3:
        maska = maska[:, :, 0]
    # Białe piksele (255) to naczynia -> 1, czarne (0) to background -> 0
    return (maska > 128).astype(np.uint8)

# ===== PRZETWARZANIE =====
def normalizuj(obraz):
    '''Normalizuje obraz do zakresu [0, 1]'''
    return (obraz - obraz.min()) / (obraz.max() - obraz.min() + 1e-8)

def zastosuj_clahe(obraz, clip_limit=0.03):
    '''Adaptive histogram equalization do preprocessing'''
    return skimage.exposure.equalize_adapthist(obraz, clip_limit=clip_limit)

def zastosuj_frangi(obraz, sigmas=range(1, 3), scale_range=(1, 6)):
    '''Filtr Frangi do detekcji naczyń'''
    return skimage.filters.frangi(obraz, sigmas=sigmas, scale_range=scale_range)

def pojasn_obraz(obraz_norm):
    '''Pojasnia obraz adaptacyjnie + power law transformation'''
    pojasn = skimage.exposure.equalize_adapthist(obraz_norm, clip_limit=0.1, nbins=256)
    return np.power(pojasn, 0.5)

def binarnizuj_otsu(obraz_norm):
    '''Binarnizuje obraz automatycznym progiem Otsu'''
    otsu_thresh = skimage.filters.threshold_otsu(obraz_norm)
    return (obraz_norm > otsu_thresh).astype(np.uint8)

def segmentuj_frangi(obraz_raw, margin=10):
    '''Pełen pipeline Frangi: preprocessing -> detekcja -> binaryzacja'''
    # Crop
    obraz_crop = obraz_raw[margin:-margin, margin:-margin]
    
    # CLAHE
    obraz_clahe = zastosuj_clahe(obraz_crop, clip_limit=CONFIG['clahe_clip'])
    
    # Frangi
    obraz_frangi = zastosuj_frangi(obraz_clahe, 
                                    sigmas=CONFIG['frangi_sigmas'],
                                    scale_range=CONFIG['frangi_scale_range'])
    
    # Normalizacja + binaryzacja
    obraz_norm = normalizuj(obraz_frangi)
    obraz_bin = binarnizuj_otsu(obraz_norm)
    
    return {
        'frangi_norm': obraz_norm,
        'frangi_pojasn': pojasn_obraz(obraz_norm),
        'segmentacja': obraz_bin,
        'n_vessels': np.sum(obraz_bin),
        'pct_vessels': np.sum(obraz_bin) / obraz_bin.size * 100,
    }

# ===== WIZUALIZACJA =====
def wyswietl_wyniki(wyniki, idx):
    '''Wyświetla 3 porównawcze obrazy przetwarzania'''
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Znormalizowany
    axes[0].imshow(wyniki['frangi_norm'], cmap='gray')
    axes[0].set_title('Frangi (znormalizowany)')
    axes[0].axis('off')
    
    # Pojaszczony
    axes[1].imshow(wyniki['frangi_pojasn'], cmap='gray')
    axes[1].set_title('Frangi (pojaszczony)')
    axes[1].axis('off')
    
    # Segmentacja
    axes[2].imshow(wyniki['segmentacja'], cmap='gray')
    axes[2].set_title(f'Segmentacja naczyń - {wyniki["pct_vessels"]:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'frangi_segmentacja_{idx+1}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Raport
    print(f"Obraz {idx+1}: {wyniki['n_vessels']} pikseli naczyń ({wyniki['pct_vessels']:.2f}%)")

def policz_metryki(y_true, y_pred):
    '''Oblicza metryki jakości segmentacji'''
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def wyswietl_porownanie(maska_expert, segmentacja, metryki, idx):
    '''Wyświetla porównanie eksperckiej maski z segmentacją Frangi'''
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Maska ekspercka
    axes[0].imshow(maska_expert, cmap='gray')
    axes[0].set_title('Maska ekspercka')
    axes[0].axis('off')
    
    # Segmentacja Frangi
    axes[1].imshow(segmentacja, cmap='gray')
    axes[1].set_title('Segmentacja Frangi')
    axes[1].axis('off')
    
    # Nałożenie (expert: zielone, Frangi: czerwone, pokrycie: żółte)
    overlay = np.zeros((*maska_expert.shape, 3))
    overlay[maska_expert == 1] = [0, 1, 0]  # Expert = zielone
    overlay[segmentacja == 1] = [1, 0, 0]   # Frangi = czerwone
    overlay[(maska_expert == 1) & (segmentacja == 1)] = [1, 1, 0]  # Pokrycie = żółte
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Nałożenie (F1={metryki["f1"]:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'porownanie_{idx+1}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Raport metryk
    print(f"\nMetryki Obraz {idx+1}:")
    print(f"  Accuracy:  {metryki['accuracy']:.3f}")
    print(f"  Precision: {metryki['precision']:.3f}")
    print(f"  Recall:    {metryki['recall']:.3f}")
    print(f"  F1-score:  {metryki['f1']:.3f}")

# ===== GŁÓWNA =====
def main():
    '''Główna funkcja: wczytaj -> przetwórz -> porównaj z expert maską'''
    obrazy_wczytane = [wczytaj_obraz(o) for o in obrazy]
    maski_wczytane = [wczytaj_maske(m) for m in maski]
    
    print("Segmentacja naczyń...")
    for idx, (obraz, maska_expert) in enumerate(zip(obrazy_wczytane, maski_wczytane)):
        # Przetwarzanie Frangi
        wyniki = segmentuj_frangi(obraz, margin=CONFIG['margin'])
        wyswietl_wyniki(wyniki, idx)
        
        # Crop maski expertу do tego samego rozmiaru (z marginesu)
        margin = CONFIG['margin']
        maska_expert_crop = maska_expert[margin:-margin, margin:-margin]
        segmentacja_crop = wyniki['segmentacja']
        
        # Resize jeśli potrzebny
        if maska_expert_crop.shape != segmentacja_crop.shape:
            maska_expert_crop = skimage.transform.resize(
                maska_expert_crop, segmentacja_crop.shape, 
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.uint8)
        
        # Oblicz metryki
        metryki = policz_metryki(maska_expert_crop.flatten(), segmentacja_crop.flatten())
        
        # Wyświetl porównanie
        wyswietl_porownanie(maska_expert_crop, segmentacja_crop, metryki, idx)
    

if __name__ == "__main__":
    main()
