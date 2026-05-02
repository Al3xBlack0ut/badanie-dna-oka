'''Segmentacja naczyń krwionośnych siatkowki - filtr Sato'''
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - tylko zapis do pliku
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
    'clahe_clip': 0.03,          # Adaptive histogram equalization clip limit 
}

# Generowanie ścieżek 
numery = ['01', '02', '03', '04', '05']  # Minimum 5 obrazów do testów
choroby = ['h']  # = 'h' - healthy, 'g' - glaucoma, 'dr' - diabetic retinopathy
obrazy = [f"{CONFIG['sciezka_img']}{num}_{ch}.jpg" for num in numery for ch in choroby]
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

def binarnizuj_otsu(obraz_norm):
    '''Binarnizuje obraz automatycznym progiem Otsu'''
    otsu_thresh = skimage.filters.threshold_otsu(obraz_norm)
    return (obraz_norm > otsu_thresh).astype(np.uint8)

def usun_artefakty(maska_bin):
    '''Post-processing: morphological closing - wypełnia przerwy w liniach naczyń'''
    # Closing = Dilation + Erosion - łączy przecięcia i wypełnia doliny
    kernel = skimage.morphology.disk(2)  # Kernel o promieniu 2 pikseli
    maska_closed = skimage.morphology.closing(maska_bin, kernel)
    return maska_closed

def segmentuj_naczynia(obraz_raw, margin=10):
    '''Pipeline segmentacji: crop -> CLAHE -> Sato -> normalizacja -> binaryzacja -> closing'''
    # Obcięcie marginesu (artefakty brzegowe)
    obraz_crop = obraz_raw[margin:-margin, margin:-margin]
    
    # Preprocessing - adaptive histogram equalization
    obraz_clahe = zastosuj_clahe(obraz_crop, clip_limit=CONFIG['clahe_clip'])
    
    # Detektor struktur tubularnych (naczynia)
    obraz_sato = zastosuj_sato(obraz_clahe, sigmas=CONFIG['sato_sigmas'])
    
    # Normalizacja do [0,1] i automatyczna binaryzacja (Otsu)
    obraz_norm = normalizuj(obraz_sato)
    obraz_bin = binarnizuj_otsu(obraz_norm)
    
    # Post-processing: morphological closing - wypełnia przerwy w liniach
    obraz_bin = usun_artefakty(obraz_bin)
    
    return {
        'segmentacja': obraz_bin,
    }

# ===== METRYKI =====
def policz_metryki(y_true, y_pred):
    '''Oblicza metryki jakości segmentacji dla niezrównoważonych klas (sklearn.metrics)'''
    # Konwersja do 1D arrays
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    
    # Macierz błędów
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Podstawowe metryki z sklearn
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)  # Sensitivity (TPR)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Miary dla danych niezrównoważonych
    sensitivity = recall  # TPR = TP / (TP + FN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR = TN / (TN + FP)
    
    # Średnia arytmetyczna - dla klasycznie niezrównoważonych danych
    balanced_accuracy_arithmetic = (sensitivity + specificity) / 2
    
    # Średnia geometryczna - dla problemu niezrównoważonych klas
    balanced_accuracy_geometric = np.sqrt(sensitivity * specificity)
    
    metryki = {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,  # recall / TPR
        'specificity': specificity,   # TNR
        'f1': f1,
        'balanced_accuracy_arithmetic': balanced_accuracy_arithmetic,
        'balanced_accuracy_geometric': balanced_accuracy_geometric,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    
    return metryki

# ===== WIZUALIZACJA =====
def wyswietl_porownanie(maska_expert, segmentacja, metryki, idx):
    '''Porównanie: expert | Sato | mapa błędów'''
    _, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Panel 1: Maska ekspercka
    axes[0].imshow(maska_expert, cmap='gray')
    axes[0].set_title('Maska ekspercka', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Segmentacja Sato
    axes[1].imshow(segmentacja, cmap='gray')
    axes[1].set_title('Segmentacja Sato', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Mapa błędów - pokazuje co gdzieś poszło źle
    # TP (zielony): Expert=1 i Sato=1
    # FP (czerwony): Expert=0 ale Sato=1
    # FN (niebieski): Expert=1 ale Sato=0
    error_map = np.zeros((*maska_expert.shape, 3))
    tp_mask = (maska_expert == 1) & (segmentacja == 1)
    fp_mask = (maska_expert == 0) & (segmentacja == 1)
    fn_mask = (maska_expert == 1) & (segmentacja == 0)
    
    error_map[tp_mask] = [0, 1, 0]      # TP = zielony
    error_map[fp_mask] = [1, 0, 0]      # FP = czerwony
    error_map[fn_mask] = [0, 0, 1]      # FN = niebieski
    
    axes[2].imshow(error_map)
    axes[2].set_title(f'Mapa błędów: TP (zielony) | FP (czerwony) | FN (niebieski)\n' +
                        f'TP={metryki["tp"]} | FP={metryki["fp"]} | FN={metryki["fn"]}\nF1={metryki["f1"]:.3f}', 
                        fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'porownanie_sato_{idx+1}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\nMetryki Obraz {idx+1}:")
    print(f"  Macierz błędów: TP={metryki['tp']}, TN={metryki['tn']}, FP={metryki['fp']}, FN={metryki['fn']}")
    print(f"  Accuracy:       {metryki['accuracy']:.3f}")
    print(f"  Sensitivity:    {metryki['sensitivity']:.3f}  (czułość, recall, TPR)")
    print(f"  Specificity:    {metryki['specificity']:.3f}  (swoistość, TNR)")
    print(f"  Precision:      {metryki['precision']:.3f}")
    print(f"  F1-score:       {metryki['f1']:.3f}")
    print(f"  Balanced Acc (AM): {metryki['balanced_accuracy_arithmetic']:.3f}  (średnia arytmetyczna)")
    print(f"  Balanced Acc (GM): {metryki['balanced_accuracy_geometric']:.3f}  (średnia geometryczna)")

# ===== GŁÓWNA =====
def main():
    '''Główny pipeline: wczytaj -> preprocess -> Sato -> porównaj z expert (minimum 5 obrazów)'''
    obrazy_wczytane = [wczytaj_obraz(o) for o in obrazy]
    maski_expert_wczytane = [wczytaj_maske_expert(m) for m in maski_expert]
    
    wyniki_wszystkie = []  # Agregacja wyników dla podsumowania
    
    print(f"Segmentacja naczyń (Sato filter) - {len(obrazy)} obrazów...\n")
    for idx, (obraz, maska_expert) in enumerate(zip(obrazy_wczytane, maski_expert_wczytane)):
        # Pipeline segmentacji
        wyniki = segmentuj_naczynia(obraz, margin=CONFIG['margin'])
        
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
        wyniki_wszystkie.append(metryki)
        
        # Wyświetl porównanie z anotacjami
        wyswietl_porownanie(maska_expert_crop, segmentacja_crop, metryki, idx)
    
    # ===== PODSUMOWANIE =====
    print("\n" + "="*70)
    print("PODSUMOWANIE - Metryki srednie dla wszystkich obrazow")
    print("="*70)
    
    keys = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 
            'balanced_accuracy_arithmetic', 'balanced_accuracy_geometric']
    labels = ['Accuracy', 'Sensitivity (czulosc)', 'Specificity (swoistosc)', 
              'Precision', 'F1-score', 'Balanced Acc (AM)', 'Balanced Acc (GM)']
    
    for key, label in zip(keys, labels):
        values = [m[key] for m in wyniki_wszystkie]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{label:30} {mean_val:.3f} +/- {std_val:.3f}")
    

if __name__ == "__main__":
    main()
