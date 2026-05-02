'''Segmentacja naczyń krwionośnych siatkowki - filtr Sato'''
import numpy as np
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import tifffile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===== KONFIGURACJA =====
CONFIG = {
    'sciezka_img': 'dno_oka/dataset/images/',
    'margin': 10,
    'sato_sigmas': range(1, 5),  # Zakresy skal dla detektora tubularnych struktur
    'clahe_clip': 0.03,          # Adaptive histogram equalization
    'adapthist_clip': 0.1,
}

# Generowanie ścieżek - numery i rodzaje obrazów
numery = ['01']
choroby = ['h']
obrazy = [f"{CONFIG['sciezka_img']}{num}_{ch}.JPG" for num in numery for ch in choroby]
maski_expert = [f"dno_oka/dataset/manual1/{num}_{ch}.tif" for num in numery for ch in choroby]  

# ===== WCZYTYWANIE =====
def wczytaj_obraz(sciezka):
    '''Wczytuje obraz JPG i konwertuje do skali szarości'''
    with open(sciezka, 'rb') as f:
        return np.array(Image.open(f).convert('L'))

def wczytaj_maske_expert(sciezka):
    '''Wczytuje anotacje eksperckie z manual1 - białe piksele (255) to naczynia'''
    maska = tifffile.imread(sciezka)
    if maska.ndim == 3:
        maska = maska[:, :, 0]
    return (maska > 128).astype(np.uint8)  # 255 -> 1 (naczynie), 0 -> 0 (background)

# ===== PRZETWARZANIE =====
def normalizuj(obraz):
    '''Normalizuje obraz do zakresu [0, 1]'''
    return (obraz - obraz.min()) / (obraz.max() - obraz.min() + 1e-8)

def zastosuj_clahe(obraz, clip_limit=0.03):
    '''Adaptive histogram equalization do preprocessing'''
    return skimage.exposure.equalize_adapthist(obraz, clip_limit=clip_limit)

def zastosuj_sato(obraz, sigmas=range(1, 5)):
    '''Filtr Sato - detektor tubularnych struktur (naczynia retinalne)'''
    return skimage.filters.sato(obraz, sigmas=sigmas, black_ridges=True)

def pojasn_obraz(obraz_norm):
    '''Pojasnia obraz adaptacyjnie + power law transformation'''
    pojasn = skimage.exposure.equalize_adapthist(obraz_norm, clip_limit=0.1, nbins=256)
    return np.power(pojasn, 0.5)

def binarnizuj_otsu(obraz_norm):
    '''Binarnizuje obraz automatycznym progiem Otsu'''
    otsu_thresh = skimage.filters.threshold_otsu(obraz_norm)
    return (obraz_norm > otsu_thresh).astype(np.uint8)

def segmentuj_naczynia(obraz_raw, margin=10):
    '''Pipeline segmentacji: crop -> CLAHE -> Sato -> normalizacja -> binaryzacja'''
    # Obcięcie marginesu (artefakty brzegowe)
    obraz_crop = obraz_raw[margin:-margin, margin:-margin]
    
    # Preprocessing - adaptive histogram equalization
    obraz_clahe = zastosuj_clahe(obraz_crop, clip_limit=CONFIG['clahe_clip'])
    
    # Detektor struktur tubularnych (naczynia)
    obraz_sato = zastosuj_sato(obraz_clahe, sigmas=CONFIG['sato_sigmas'])
    
    # Normalizacja do [0,1] i automatyczna binaryzacja (Otsu)
    obraz_norm = normalizuj(obraz_sato)
    obraz_bin = binarnizuj_otsu(obraz_norm)
    
    return {
        'sato_norm': obraz_norm,
        'sato_pojasn': pojasn_obraz(obraz_norm),
        'segmentacja': obraz_bin,
        'n_vessels': np.sum(obraz_bin),
        'pct_vessels': np.sum(obraz_bin) / obraz_bin.size * 100,
    }

# ===== WIZUALIZACJA =====
def wyswietl_wyniki(wyniki, idx):
    '''Wyświetla przetwarzanie: Sato znormalizowany, pojaśniony i binaryzację'''
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sato znormalizowany
    axes[0].imshow(wyniki['sato_norm'], cmap='gray')
    axes[0].set_title('Sato (znormalizowany)')
    axes[0].axis('off')
    
    # Sato pojaśniony
    axes[1].imshow(wyniki['sato_pojasn'], cmap='gray')
    axes[1].set_title('Sato (pojaśniony)')
    axes[1].axis('off')
    
    # Binaryzacja
    axes[2].imshow(wyniki['segmentacja'], cmap='gray')
    axes[2].set_title(f'Segmentacja - {wyniki["pct_vessels"]:.1f}% pikseli')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sato_segmentacja_{idx+1}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Obraz {idx+1}: {wyniki['n_vessels']} pikseli naczyń ({wyniki['pct_vessels']:.2f}%)")

def policz_metryki(y_true, y_pred):
    '''Oblicza metryki jakości segmentacji (sklearn.metrics)'''
    # Konwersja do 1D arrays
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    
    # Metryki z sklearn
    metryki = {
        'accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
        'recall': recall_score(y_true_flat, y_pred_flat, zero_division=0),
        'f1': f1_score(y_true_flat, y_pred_flat, zero_division=0),
    }
    
    # Macierz błędów
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metryki.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    
    return metryki

def wyswietl_porownanie(maska_expert, segmentacja, metryki, idx):
    '''Porównanie: expert (zielony) vs Sato (czerwony) vs pokrycie (żółty)'''
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anotacje eksperckie
    axes[0].imshow(maska_expert, cmap='gray')
    axes[0].set_title('Anotacje eksperckie')
    axes[0].axis('off')
    
    # Segmentacja Sato
    axes[1].imshow(segmentacja, cmap='gray')
    axes[1].set_title('Segmentacja Sato')
    axes[1].axis('off')
    
    # Nałożenie RGB
    overlay = np.zeros((*maska_expert.shape, 3))
    overlay[maska_expert == 1] = [0, 1, 0]  # Expert = zielony
    overlay[segmentacja == 1] = [1, 0, 0]   # Sato = czerwony
    overlay[(maska_expert == 1) & (segmentacja == 1)] = [1, 1, 0]  # Pokrycie = żółty
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Porównanie (F1={metryki["f1"]:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'porownanie_sato_{idx+1}.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\nMetryki Obraz {idx+1}:")
    print(f"  Accuracy:  {metryki['accuracy']:.3f}")
    print(f"  Precision: {metryki['precision']:.3f}")
    print(f"  Recall:    {metryki['recall']:.3f}")
    print(f"  F1-score:  {metryki['f1']:.3f}")

# ===== GŁÓWNA =====
def main():
    '''Główny pipeline: wczytaj -> preprocess -> Sato -> porównaj z expert'''
    obrazy_wczytane = [wczytaj_obraz(o) for o in obrazy]
    maski_expert_wczytane = [wczytaj_maske_expert(m) for m in maski_expert]
    
    print("Segmentacja naczyń (Sato filter)...")
    for idx, (obraz, maska_expert) in enumerate(zip(obrazy_wczytane, maski_expert_wczytane)):
        # Pipeline segmentacji
        wyniki = segmentuj_naczynia(obraz, margin=CONFIG['margin'])
        wyswietl_wyniki(wyniki, idx)
        
        # Obcięcie maski eksperta do tego samego rozmiaru (bez marginesu)
        margin = CONFIG['margin']
        maska_expert_crop = maska_expert[margin:-margin, margin:-margin]
        segmentacja_crop = wyniki['segmentacja']
        
        # Resize jeśli potrzebny (wyrównanie wymiarów)
        if maska_expert_crop.shape != segmentacja_crop.shape:
            maska_expert_crop = skimage.transform.resize(
                maska_expert_crop, segmentacja_crop.shape, 
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.uint8)
        
        # Oblicz metryki segmentacji
        metryki = policz_metryki(maska_expert_crop.flatten(), segmentacja_crop.flatten())
        
        # Wyświetl porównanie z anotacjami
        wyswietl_porownanie(maska_expert_crop, segmentacja_crop, metryki, idx)
    

if __name__ == "__main__":
    main()
