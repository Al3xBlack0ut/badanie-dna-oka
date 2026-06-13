'''Segmentacja naczyń krwionośnych siatkówki: Sato, Random Forest i CNN.'''
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'medycynie_matplotlib'))

import matplotlib.pyplot as plt
import numpy as np
import skimage
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ===== KONFIGURACJA =====
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / 'dataset'

CONFIG = {
    'margin': 0,
    'patch_size': 5,
    'sato_sigmas': range(1, 5),
    'clahe_clip': 0.03,
    'random_state': 13,
    'max_samples_per_class_per_image': 4500,
    'prediction_chunk_size': 150000,
    'cnn_patch_size': 17,
    'cnn_epochs': 10,
    'cnn_batch_size': 768,
    'cnn_learning_rate': 0.0007,
    'cnn_weight_decay': 0.0001,
    'cnn_prediction_chunk_size': 50000,
    'cnn_validation_fraction': 0.20,
    'cnn_max_positive_samples_per_image': 3200,
    'cnn_negative_to_positive_ratio': 2.5,
    'cnn_threshold_min': 0.40,
    'cnn_threshold_max': 0.95,
}

RESULTS_DIR = BASE_DIR / 'wyniki'
COMPARISON_DIR = RESULTS_DIR / 'porownania'
MASKS_DIR = RESULTS_DIR / 'maski'

# Hold-out: klasyfikatory uczą się na 5 pierwszych obrazach każdego typu,
# a testowane są na 2 ostatnich obrazach każdego typu.
TRAIN_IDS = [
    f'{idx:02d}_{wariant}'
    for wariant in ['h', 'g', 'dr']
    for idx in range(1, 6)
]
HOLDOUT_TEST_IDS = [
    f'{idx:02d}_{wariant}'
    for wariant in ['h', 'g', 'dr']
    for idx in [14, 15]
]


def sciezka_obrazu(image_id):
    '''Zwraca ścieżkę do pliku obrazu, uwzględniając rozszerzenia jpg/JPG.'''
    for ext in ('.jpg', '.JPG', '.jpeg', '.JPEG'):
        path = DATASET_DIR / 'images' / f'{image_id}{ext}'
        if path.exists():
            return path
    raise FileNotFoundError(f'Brak obrazu dla identyfikatora: {image_id}')


def sciezka_maski_expert(image_id):
    '''Zwraca ścieżkę do pliku maski eksperckiej, uwzględniając rozszerzenia tif/TIF.'''
    return DATASET_DIR / 'manual1' / f'{image_id}.tif'


def sciezka_maski_pola(image_id):
    '''Zwraca ścieżkę do pliku maski pola, uwzględniając rozszerzenia tif/TIF.'''
    return DATASET_DIR / 'mask' / f'{image_id}_mask.tif'


# ===== WCZYTYWANIE =====
def wczytaj_obraz_rgb(sciezka):
    '''Wczytuje obraz RGB i normalizuje go do zakresu [0, 1].'''
    with Image.open(sciezka) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def wczytaj_obraz(sciezka):
    '''Wczytuje obraz w skali szarości dla klasycznego pipeline Sato.'''
    with Image.open(sciezka) as image:
        return np.asarray(image.convert('L'), dtype=np.float32) / 255.0


def wczytaj_maske(sciezka):
    '''Wczytuje maskę binarną: piksele jasne oznaczają klasę pozytywną.'''
    maska = tifffile.imread(sciezka)
    if maska.ndim == 3:
        maska = maska[:, :, 0]
    return (maska > 128).astype(np.uint8)


def przytnij_margines(obraz, margin):
    '''Przycina margines z obrazu, jeśli margin > 0.'''
    if margin == 0:
        return obraz
    return obraz[margin:-margin, margin:-margin]


# ===== PRZETWARZANIE OBRAZU =====
def normalizuj(obraz):
    '''Normalizuje obraz do zakresu [0, 1].'''
    return (obraz - obraz.min()) / (obraz.max() - obraz.min() + 1e-8)


def zastosuj_clahe(obraz, clip_limit=0.03):
    '''Adaptive histogram equalization do wstępnego przetwarzania.'''
    return skimage.exposure.equalize_adapthist(obraz, clip_limit=clip_limit)


def zastosuj_sato(obraz, sigmas=range(1, 5)):
    '''Filtr Sato - detektor struktur tubularnych podobnych do naczyń.'''
    return skimage.filters.sato(obraz, sigmas=sigmas, black_ridges=True)


def binarnizuj_otsu(obraz_norm):
    '''Binaryzuje obraz automatycznym progiem Otsu.'''
    otsu_thresh = skimage.filters.threshold_otsu(obraz_norm)
    return (obraz_norm > otsu_thresh).astype(np.uint8)


def usun_artefakty(maska_bin, maska_pola=None):
    '''Post-processing: domknięcie morfologiczne i usunięcie drobnych obiektów.'''
    kernel = skimage.morphology.disk(2)
    maska = skimage.morphology.closing(maska_bin.astype(bool), kernel)
    maska = skimage.morphology.remove_small_objects(maska, max_size=19)
    if maska_pola is not None:
        maska &= maska_pola.astype(bool)
    return maska.astype(np.uint8)


def segmentuj_naczynia(obraz_raw, maska_pola=None, margin=0):
    '''Pipeline: CLAHE -> Sato -> normalizacja -> Otsu -> morfologia.'''
    obraz_crop = przytnij_margines(obraz_raw, margin)
    maska_pola_crop = przytnij_margines(maska_pola, margin) if maska_pola is not None else None

    obraz_clahe = zastosuj_clahe(obraz_crop, clip_limit=CONFIG['clahe_clip'])
    obraz_sato = zastosuj_sato(obraz_clahe, sigmas=CONFIG['sato_sigmas'])
    obraz_norm = normalizuj(obraz_sato)
    obraz_bin = binarnizuj_otsu(obraz_norm)
    obraz_bin = usun_artefakty(obraz_bin, maska_pola_crop)

    return {
        'preprocessed': obraz_clahe,
        'sato': obraz_norm,
        'segmentacja': obraz_bin,
    }


# ===== CECHY 5x5 I KLASYFIKATOR =====
def przygotuj_kanaly_cech(rgb_crop):
    '''Tworzy kanały, z których będą liczone cechy okien 5x5.'''
    red = rgb_crop[:, :, 0]
    green = rgb_crop[:, :, 1]
    blue = rgb_crop[:, :, 2]
    green_clahe = zastosuj_clahe(green, clip_limit=CONFIG['clahe_clip'])
    sato = normalizuj(zastosuj_sato(green_clahe, sigmas=CONFIG['sato_sigmas']))

    return {
        'red': red,
        'green': green,
        'blue': blue,
        'green_clahe': green_clahe,
        'sato': sato,
    }


def wyznacz_cechy_z_okien(kanaly, coords, patch_size=5):
    '''
    Ekstrakcja cech dla wycinków patch_size x patch_size.
    Decyzja klasy dotyczy środkowego piksela wycinka.
    '''
    coords = np.asarray(coords, dtype=np.int64)
    pad = patch_size // 2
    rr = coords[:, 0]
    cc = coords[:, 1]
    cechy = []

    grid = np.arange(-pad, pad + 1, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid)

    for nazwa, kanal in kanaly.items():
        padded = np.pad(kanal, pad_width=pad, mode='reflect')
        windows = skimage.util.view_as_windows(padded, (patch_size, patch_size))
        patches = windows[rr, cc].astype(np.float32)

        mean = patches.mean(axis=(1, 2))
        centered = patches - mean[:, None, None]

        cechy.extend([
            patches[:, pad, pad],
            mean,
            patches.std(axis=(1, 2)),
            patches.var(axis=(1, 2)),
            patches.min(axis=(1, 2)),
            patches.max(axis=(1, 2)),
            (patches.max(axis=(1, 2)) - patches.min(axis=(1, 2))),
        ])

        if nazwa in ('green_clahe', 'sato'):
            cechy.extend([
                (centered * xx).mean(axis=(1, 2)),
                (centered * yy).mean(axis=(1, 2)),
                (patches * xx * xx).mean(axis=(1, 2)),
                (patches * yy * yy).mean(axis=(1, 2)),
                (patches * xx * yy).mean(axis=(1, 2)),
            ])

    return np.column_stack(cechy).astype(np.float32)


def losuj_wspolrzedne(mask_pos, mask_valid, limit, rng):
    '''Losuje współrzędne pikseli z maski pozytywnej, ograniczając do limitu.'''
    coords = np.column_stack(np.where(mask_pos & mask_valid))
    if len(coords) > limit:
        coords = coords[rng.choice(len(coords), size=limit, replace=False)]
    return coords


def zbuduj_zbior_uczacy(image_ids):
    '''Buduje zbalansowany zbiór uczący z losowo wybranych wycinków 5x5.'''
    rng = np.random.default_rng(CONFIG['random_state'])
    x_parts = []
    y_parts = []

    for image_id in image_ids:
        rgb = przytnij_margines(wczytaj_obraz_rgb(sciezka_obrazu(image_id)), CONFIG['margin'])
        expert = przytnij_margines(wczytaj_maske(sciezka_maski_expert(image_id)), CONFIG['margin']).astype(bool)
        maska_pola = przytnij_margines(wczytaj_maske(sciezka_maski_pola(image_id)), CONFIG['margin']).astype(bool)

        kanaly = przygotuj_kanaly_cech(rgb)
        limit = CONFIG['max_samples_per_class_per_image']
        pos_coords = losuj_wspolrzedne(expert, maska_pola, limit, rng)
        neg_coords = losuj_wspolrzedne(~expert, maska_pola, len(pos_coords), rng)

        coords = np.vstack([pos_coords, neg_coords])
        y = np.concatenate([
            np.ones(len(pos_coords), dtype=np.uint8),
            np.zeros(len(neg_coords), dtype=np.uint8),
        ])

        order = rng.permutation(len(coords))
        coords = coords[order]
        y = y[order]

        x_parts.append(wyznacz_cechy_z_okien(kanaly, coords, CONFIG['patch_size']))
        y_parts.append(y)
        print(f'  {image_id}: próbki uczące={len(y)} (naczynia={int(y.sum())}, tło={int((y == 0).sum())})')

    return np.vstack(x_parts), np.concatenate(y_parts)


def trenuj_klasyfikator(image_ids):
    '''Trenuje prosty klasyfikator scikit-learn na cechach z okien 5x5.'''
    print('\nTrening klasyfikatora RandomForest na cechach 5x5...')
    x_train, y_train = zbuduj_zbior_uczacy(image_ids)

    klasyfikator = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=CONFIG['random_state'],
    )
    klasyfikator.fit(x_train, y_train)
    print(f'  Razem próbek: {len(y_train)}, liczba cech: {x_train.shape[1]}')
    return klasyfikator


def predykcja_klasyfikatora(klasyfikator, rgb_raw, maska_pola_raw, margin=0):
    '''Predykcja maski naczyń dla całego obrazu z użyciem klasyfikatora.'''
    rgb = przytnij_margines(rgb_raw, margin)
    maska_pola = przytnij_margines(maska_pola_raw, margin).astype(bool)
    kanaly = przygotuj_kanaly_cech(rgb)

    coords = np.column_stack(np.where(maska_pola))
    segmentacja = np.zeros(maska_pola.shape, dtype=np.uint8)

    chunk = CONFIG['prediction_chunk_size']
    for start in range(0, len(coords), chunk):
        coords_chunk = coords[start:start + chunk]
        x_chunk = wyznacz_cechy_z_okien(kanaly, coords_chunk, CONFIG['patch_size'])
        segmentacja[coords_chunk[:, 0], coords_chunk[:, 1]] = klasyfikator.predict(x_chunk)

    return usun_artefakty(segmentacja, maska_pola)


# ===== GŁĘBOKA SIEĆ NEURONOWA CNN =====
def wyznacz_patche_cnn(kanaly, coords, patch_size=None):
    '''Tworzy tensory patchy dla sieci CNN z kanałów green_clahe i sato.'''
    patch_size = CONFIG['cnn_patch_size'] if patch_size is None else patch_size
    coords = np.asarray(coords, dtype=np.int64)
    pad = patch_size // 2
    rr = coords[:, 0]
    cc = coords[:, 1]
    patches_all = []

    for nazwa in ('green_clahe', 'sato'):
        kanal = kanaly[nazwa]
        padded = np.pad(kanal, pad_width=pad, mode='reflect')
        windows = skimage.util.view_as_windows(padded, (patch_size, patch_size))
        patches_all.append(windows[rr, cc].astype(np.float32))

    return np.stack(patches_all, axis=1)


def zbuduj_zbior_uczacy_cnn(image_ids):
    '''Buduje zbiór uczący patchy dla CNN z większą liczbą przykładów tła.'''
    rng = np.random.default_rng(CONFIG['random_state'])
    x_parts = []
    y_parts = []

    for image_id in image_ids:
        rgb = przytnij_margines(wczytaj_obraz_rgb(sciezka_obrazu(image_id)), CONFIG['margin'])
        expert = przytnij_margines(wczytaj_maske(sciezka_maski_expert(image_id)), CONFIG['margin']).astype(bool)
        maska_pola = przytnij_margines(wczytaj_maske(sciezka_maski_pola(image_id)), CONFIG['margin']).astype(bool)

        kanaly = przygotuj_kanaly_cech(rgb)
        pos_limit = CONFIG['cnn_max_positive_samples_per_image']
        pos_coords = losuj_wspolrzedne(expert, maska_pola, pos_limit, rng)
        neg_limit = int(len(pos_coords) * CONFIG['cnn_negative_to_positive_ratio'])
        neg_coords = losuj_wspolrzedne(~expert, maska_pola, neg_limit, rng)

        coords = np.vstack([pos_coords, neg_coords])
        y = np.concatenate([
            np.ones(len(pos_coords), dtype=np.int64),
            np.zeros(len(neg_coords), dtype=np.int64),
        ])

        order = rng.permutation(len(coords))
        coords = coords[order]
        y = y[order]

        x_parts.append(wyznacz_patche_cnn(kanaly, coords, CONFIG['cnn_patch_size']))
        y_parts.append(y)
        print(f'  {image_id}: próbki CNN={len(y)} (naczynia={int(y.sum())}, tło={int((y == 0).sum())})')

    return np.vstack(x_parts), np.concatenate(y_parts)


class MalaSiecCNN(nn.Module):
    '''Prosta głęboka sieć CNN klasyfikująca środkowy piksel patcha.'''

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        '''Przekazuje dane przez sieć, zwracając logity dla dwóch klas.'''
        return self.model(x)


def wybierz_urzadzenie_torch():
    '''Wybiera MPS na Apple Silicon, jeśli jest dostępne, w innym razie CPU.'''
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def policz_prawdopodobienstwa_cnn(model, loader, device):
    '''Zwraca prawdopodobieństwa klasy naczynia i etykiety dla podanego loadera.'''
    probs_parts = []
    y_parts = []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_parts.append(probs)
            y_parts.append(y_batch.numpy())

    return np.concatenate(probs_parts), np.concatenate(y_parts)


def dobierz_prog_cnn(model, loader, device):
    '''Dobiera próg decyzji CNN na walidacji, maksymalizując F1.'''
    probs, y_true = policz_prawdopodobienstwa_cnn(model, loader, device)
    najlepszy_prog = 0.5
    najlepszy_f1 = -1.0

    for prog in np.linspace(CONFIG['cnn_threshold_min'], CONFIG['cnn_threshold_max'], 56):
        y_pred = (probs >= prog).astype(np.uint8)
        wynik_f1 = f1_score(y_true, y_pred, zero_division=0)
        if wynik_f1 > najlepszy_f1:
            najlepszy_f1 = wynik_f1
            najlepszy_prog = float(prog)

    return najlepszy_prog, najlepszy_f1


def trenuj_cnn(image_ids):
    '''Trenuje głęboką sieć CNN PyTorch na patchach obrazu.'''
    print('\nTrening głębokiej sieci CNN PyTorch na większym zbiorze patchy obrazu...')
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])

    x_train, y_train = zbuduj_zbior_uczacy_cnn(image_ids)
    rng = np.random.default_rng(CONFIG['random_state'])
    order = rng.permutation(len(y_train))
    val_size = int(len(y_train) * CONFIG['cnn_validation_fraction'])
    val_idx = order[:val_size]
    train_idx = order[val_size:]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train[train_idx]).float(),
        torch.from_numpy(y_train[train_idx]).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_train[val_idx]).float(),
        torch.from_numpy(y_train[val_idx]).long(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['cnn_batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['cnn_batch_size'],
        shuffle=False,
    )

    device = wybierz_urzadzenie_torch()
    model = MalaSiecCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['cnn_learning_rate'],
        weight_decay=CONFIG['cnn_weight_decay'],
    )

    print(
        f'  Razem próbek: {len(y_train)}, trening={len(train_dataset)}, '
        f'walidacja={len(val_dataset)}, kształt wejścia: {x_train.shape[1:]}, '
        f'urządzenie: {device}, lr={CONFIG["cnn_learning_rate"]}'
    )
    for epoch in range(CONFIG['cnn_epochs']):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)

        val_probs, val_true = policz_prawdopodobienstwa_cnn(model, val_loader, device)
        val_pred = (val_probs >= 0.5).astype(np.uint8)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        print(
            f'  Epoka {epoch + 1}/{CONFIG["cnn_epochs"]}: '
            f'loss={total_loss / total:.4f}, accuracy={correct / total:.3f}, val_f1@0.50={val_f1:.3f}'
        )

    prog, val_f1 = dobierz_prog_cnn(model, val_loader, device)
    model.prog_decyzji = prog
    print(f'  Wybrany próg decyzji CNN: {prog:.2f} (F1 walidacyjne={val_f1:.3f})')
    return model


def predykcja_cnn(model, rgb_raw, maska_pola_raw, margin=0):
    '''Predykcja maski naczyń dla całego obrazu z użyciem głębokiej sieci CNN.'''
    rgb = przytnij_margines(rgb_raw, margin)
    maska_pola = przytnij_margines(maska_pola_raw, margin).astype(bool)
    kanaly = przygotuj_kanaly_cech(rgb)

    coords = np.column_stack(np.where(maska_pola))
    segmentacja = np.zeros(maska_pola.shape, dtype=np.uint8)
    device = next(model.parameters()).device
    chunk = CONFIG['cnn_prediction_chunk_size']

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), chunk):
            coords_chunk = coords[start:start + chunk]
            x_chunk = wyznacz_patche_cnn(kanaly, coords_chunk, CONFIG['cnn_patch_size'])
            x_tensor = torch.from_numpy(x_chunk).float().to(device)
            probs = torch.softmax(model(x_tensor), dim=1)[:, 1].cpu().numpy()
            predictions = (probs >= model.prog_decyzji).astype(np.uint8)
            segmentacja[coords_chunk[:, 0], coords_chunk[:, 1]] = predictions

    return usun_artefakty(segmentacja, maska_pola)


# ===== METRYKI =====
def policz_metryki(y_true, y_pred, valid_mask=None):
    '''Oblicza metryki jakości segmentacji dla niezrównoważonych klas.'''
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    if valid_mask is not None:
        valid_flat = np.asarray(valid_mask).astype(bool).flatten()
        y_true_flat = y_true_flat[valid_flat]
        y_pred_flat = y_pred_flat[valid_flat]

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy_arithmetic = (sensitivity + specificity) / 2
    balanced_accuracy_geometric = np.sqrt(sensitivity * specificity)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'balanced_accuracy_arithmetic': balanced_accuracy_arithmetic,
        'balanced_accuracy_geometric': balanced_accuracy_geometric,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'confusion_matrix': cm,
    }


def wypisz_metryki(metryki, label):
    '''wypisuje metryki w czytelnej formie, wraz z macierzą pomyłek i liczbą TP/FP/FN/TN.'''
    tn, fp, fn, tp = metryki['tn'], metryki['fp'], metryki['fn'], metryki['tp']
    print(f'\nMetryki {label}:')
    print('  Macierz pomyłek [[TN, FP], [FN, TP]]:')
    print(f'  [[{tn}  {fp}] [{fn}  {tp}]]')
    print(f"  TP={metryki['tp']}, TN={metryki['tn']}, FP={metryki['fp']}, FN={metryki['fn']}")
    print(f"  Accuracy:          {metryki['accuracy']:.3f}")
    print(f"  Sensitivity:       {metryki['sensitivity']:.3f}")
    print(f"  Specificity:       {metryki['specificity']:.3f}")
    print(f"  Precision:         {metryki['precision']:.3f}")
    print(f"  F1-score:          {metryki['f1']:.3f}")
    print(f"  Balanced Acc (AM): {metryki['balanced_accuracy_arithmetic']:.3f}")
    print(f"  Balanced Acc (GM): {metryki['balanced_accuracy_geometric']:.3f}")


# ===== WIZUALIZACJA =====
def naloz_segmentacje(rgb_crop, segmentacja):
    '''Zamalowuje piksele sklasyfikowane jako naczynia na czerwono.'''
    overlay = rgb_crop.copy()
    vessel = segmentacja.astype(bool)
    overlay[vessel] = 0.45 * overlay[vessel] + 0.55 * np.array([1.0, 0.0, 0.0])
    return np.clip(overlay, 0, 1)


def zapisz_maske_binarnie(segmentacja, image_id, metoda):
    '''Zapisuje binarną maskę odpowiedzi algorytmu jako PNG 0/255.'''
    output_dir = MASKS_DIR / metoda
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{image_id}_maska.png'
    Image.fromarray((segmentacja.astype(np.uint8) * 255)).save(output_path)


def wyswietl_porownanie(rgb_crop, maska_expert, segmentacja, metryki, image_id, prefix, tytul):
    '''Zapisuje: maska ekspert | porównanie segmentacji z maską ekspercką.'''
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(maska_expert, cmap='gray')
    axes[0].set_title('Maska ekspercka', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    error_map = np.zeros((*maska_expert.shape, 3))
    tp_mask = (maska_expert == 1) & (segmentacja == 1)
    fp_mask = (maska_expert == 0) & (segmentacja == 1)
    fn_mask = (maska_expert == 1) & (segmentacja == 0)

    error_map[tp_mask] = [0, 1, 0]
    error_map[fp_mask] = [1, 0, 0]
    error_map[fn_mask] = [0, 0, 1]

    axes[1].imshow(error_map)
    axes[1].set_title(
        f'TP zielony | FP czerwony | FN niebieski\n'
        f'TP={metryki["tp"]} | FP={metryki["fp"]} | FN={metryki["fn"]}\n'
        f'{tytul} | F1={metryki["f1"]:.3f}',
        fontsize=11,
        fontweight='bold',
    )
    axes[1].axis('off')

    plt.tight_layout()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(COMPARISON_DIR / f'{prefix}_{image_id}.png', dpi=100, bbox_inches='tight')
    plt.close()


def podsumuj_wyniki(wyniki_wszystkie, naglowek):
    '''Podsumowuje wyniki dla wielu obrazów, wypisuje średnie i odchylenia standardowe metryk.'''
    print('\n' + '=' * 70)
    print(naglowek)
    print('=' * 70)

    keys = [
        'accuracy',
        'sensitivity',
        'specificity',
        'precision',
        'f1',
        'balanced_accuracy_arithmetic',
        'balanced_accuracy_geometric',
    ]
    labels = [
        'Accuracy',
        'Sensitivity',
        'Specificity',
        'Precision',
        'F1-score',
        'Balanced Acc (AM)',
        'Balanced Acc (GM)',
    ]

    for key, label in zip(keys, labels):
        values = [m[key] for m in wyniki_wszystkie]
        print(f'{label:30} {np.mean(values):.3f} +/- {np.std(values):.3f}')


# ===== GŁÓWNE =====
def uruchom_sato():
    '''Uruchamia klasyczny algorytm przetwarzania obrazu na zbiorze hold-out.'''
    wyniki_wszystkie = []
    print(f'Segmentacja naczyń filtrem Sato na hold-out - {len(HOLDOUT_TEST_IDS)} obrazów...')

    for image_id in HOLDOUT_TEST_IDS:
        rgb = wczytaj_obraz_rgb(sciezka_obrazu(image_id))
        obraz = wczytaj_obraz(sciezka_obrazu(image_id))
        maska_expert = wczytaj_maske(sciezka_maski_expert(image_id))
        maska_pola = wczytaj_maske(sciezka_maski_pola(image_id))

        wyniki = segmentuj_naczynia(obraz, maska_pola, margin=CONFIG['margin'])
        rgb_crop = przytnij_margines(rgb, CONFIG['margin'])
        maska_expert_crop = przytnij_margines(maska_expert, CONFIG['margin'])
        maska_pola_crop = przytnij_margines(maska_pola, CONFIG['margin'])
        segmentacja_crop = wyniki['segmentacja']

        metryki = policz_metryki(maska_expert_crop, segmentacja_crop, maska_pola_crop)
        wyniki_wszystkie.append(metryki)
        wypisz_metryki(metryki, f'Sato {image_id}')
        zapisz_maske_binarnie(segmentacja_crop, image_id, 'sato')
        wyswietl_porownanie(
            rgb_crop,
            maska_expert_crop,
            segmentacja_crop,
            metryki,
            image_id,
            'porownanie_sato',
            'Segmentacja Sato',
        )

    podsumuj_wyniki(wyniki_wszystkie, 'PODSUMOWANIE SATO - niezależny zbiór hold-out')


def uruchom_random_forest():
    '''Trenuje klasyfikator na cechach 5x5 i testuje go na niezależnym hold-out.'''
    klasyfikator = trenuj_klasyfikator(TRAIN_IDS)
    wyniki_wszystkie = []

    print(f'\nPredykcja klasyfikatora na hold-out - {len(HOLDOUT_TEST_IDS)} obrazów...')
    for image_id in HOLDOUT_TEST_IDS:
        rgb = wczytaj_obraz_rgb(sciezka_obrazu(image_id))
        maska_expert = wczytaj_maske(sciezka_maski_expert(image_id))
        maska_pola = wczytaj_maske(sciezka_maski_pola(image_id))

        segmentacja = predykcja_klasyfikatora(klasyfikator, rgb, maska_pola, margin=CONFIG['margin'])
        rgb_crop = przytnij_margines(rgb, CONFIG['margin'])
        maska_expert_crop = przytnij_margines(maska_expert, CONFIG['margin'])
        maska_pola_crop = przytnij_margines(maska_pola, CONFIG['margin'])

        metryki = policz_metryki(maska_expert_crop, segmentacja, maska_pola_crop)
        wyniki_wszystkie.append(metryki)
        wypisz_metryki(metryki, f'RandomForest 5x5 {image_id}')
        zapisz_maske_binarnie(segmentacja, image_id, 'random_forest')
        wyswietl_porownanie(
            rgb_crop,
            maska_expert_crop,
            segmentacja,
            metryki,
            image_id,
            'porownanie_rf',
            'RandomForest, cechy 5x5',
        )

    podsumuj_wyniki(
        wyniki_wszystkie,
        'PODSUMOWANIE RANDOM FOREST 5x5 - niezależny zbiór hold-out',
    )


def uruchom_cnn():
    '''Trenuje głęboką sieć CNN i testuje ją na niezależnym hold-out.'''
    model = trenuj_cnn(TRAIN_IDS)
    wyniki_wszystkie = []

    print(f'\nPredykcja CNN na hold-out - {len(HOLDOUT_TEST_IDS)} obrazów...')
    for image_id in HOLDOUT_TEST_IDS:
        rgb = wczytaj_obraz_rgb(sciezka_obrazu(image_id))
        maska_expert = wczytaj_maske(sciezka_maski_expert(image_id))
        maska_pola = wczytaj_maske(sciezka_maski_pola(image_id))

        segmentacja = predykcja_cnn(model, rgb, maska_pola, margin=CONFIG['margin'])
        rgb_crop = przytnij_margines(rgb, CONFIG['margin'])
        maska_expert_crop = przytnij_margines(maska_expert, CONFIG['margin'])
        maska_pola_crop = przytnij_margines(maska_pola, CONFIG['margin'])

        metryki = policz_metryki(maska_expert_crop, segmentacja, maska_pola_crop)
        wyniki_wszystkie.append(metryki)
        wypisz_metryki(metryki, f'CNN PyTorch {CONFIG["cnn_patch_size"]}x{CONFIG["cnn_patch_size"]} {image_id}')
        zapisz_maske_binarnie(segmentacja, image_id, 'cnn')
        wyswietl_porownanie(
            rgb_crop,
            maska_expert_crop,
            segmentacja,
            metryki,
            image_id,
            'porownanie_cnn',
            f'CNN PyTorch, patche {CONFIG["cnn_patch_size"]}x{CONFIG["cnn_patch_size"]}',
        )

    podsumuj_wyniki(
        wyniki_wszystkie,
        f'PODSUMOWANIE CNN PYTORCH {CONFIG["cnn_patch_size"]}x{CONFIG["cnn_patch_size"]} - niezależny zbiór hold-out',
    )


def main():
    '''main'''
    uruchom_sato()
    uruchom_random_forest()
    uruchom_cnn()


if __name__ == '__main__':
    main()
