# Segmentacja naczyń krwionośnych dna oka

Projekt porównuje trzy metody segmentacji naczyń siatkówki na obrazach z bazy HRF:

- klasyczny pipeline `CLAHE -> filtr Sato -> Otsu -> morfologia`,
- `RandomForestClassifier` na ręcznie liczonych cechach z okien `5x5`,
- małą sieć CNN w PyTorch klasyfikującą środkowy piksel patcha `17x17`.

## Dane

Katalog `dataset` ma trzy podkatalogi:

- `images` - kolorowe fotografie dna oka,
- `manual1` - eksperckie maski naczyń,
- `mask` - maski pola widzenia oka.

Metryki są liczone tylko w masce pola widzenia, żeby czarne tło poza siatkówką nie zawyżało wyników.

## Podział danych

Modele Random Forest i CNN uczą się na obrazach `01-05` dla wariantów `h`, `g`, `dr`. Test odbywa się na niezależnym hold-oucie: `14_h`, `15_h`, `14_g`, `15_g`, `14_dr`, `15_dr`. Metoda Sato też jest oceniana na tym samym hold-oucie.

## Uruchomienie

Notebook z raportem:

```bash
jupyter notebook raport_dno_oka.ipynb
```

Skrypt:

```bash
python dno_oka.py
```

Skrypt zapisuje maski i porównania do katalogu `wyniki`.

## Wyniki z raportu

Średnie wyniki na hold-oucie:

| Metoda | Sensitivity | Specificity | Precision | F1 | Balanced Acc |
|---|---:|---:|---:|---:|---:|
| Sato | 0.640 | 0.930 | 0.532 | 0.563 | 0.785 |
| Random Forest 5x5 | 0.890 | 0.916 | 0.523 | 0.653 | 0.903 |
| CNN 17x17 | 0.818 | 0.948 | 0.617 | 0.698 | 0.883 |

Sato jest prostym i interpretowalnym baseline'em. Random Forest ma najwyższą czułość, więc wykrywa najwięcej naczyń. CNN daje najlepsze F1 i precyzję, czyli robi mniej fałszywych oznaczeń tła jako naczyń.

