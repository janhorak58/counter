# Obrazová detekce návštěvníků na Jizerské magistrále

- tento repozitář slouží jako podpora bakalářské práce zaměřené na detekci a počítání návštěvníků na Jizerské magistrále pomocí počítačového vidění a strojového učení.

Systém pro automatické počítání průchodů objektů přes definované čáry ve videu pomocí YOLO detekce a trackingu.

## Funkce

- Detekce a tracking objektů (turisté, lyžaři, cyklisté, psi)
- Počítání průchodů přes jednu nebo více čar
- Rozlišení směru (IN/OUT)
- Vizualizace trajektorií a počtů
- Export anotovaného videa

## Instalace
```bash
pip install -r requirements.txt
```

## Třídy objektů

| ID | Třída | Barva |
|----|-------|-------|
| 0 | tourist | Zelená |
| 1 | skier | Červená |
| 2 | cyclist | Modrá |
| 3 | tourist_dog | Žlutá |

## Struktura projektu
```
counter/
├── src/
│   ├── main.py
│   └── models/
│       ├── Counter.py
│       ├── LineCounter.py
│       ├── ObjectTracker.py
│       └── DetectedObject.py
├── models/
│   └── yolov5n_v2/
│           └── weights/
│               └── best.pt
└── data/
    └── videos/
```

## Použití
```bash
python -m src.main <video_filename>
```

nebo bez argumentu pro výchozí video:
```bash
python -m src.main
```
