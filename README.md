# Counter – video line-crossing object counter (YOLO / RF-DETR)

Počítá průchody objektů přes zadanou čáru ve videu (IN/OUT). Pipeline: **detekce → tracking → line-crossing logika → counts.json + volitelně video export**.

## Kanonické třídy (výstup)
Ve výstupech a evaluaci se používají fixní ID:

- `0` = `TOURIST`
- `1` = `SKIER`
- `2` = `CYCLIST`
- `3` = `TOURIST_DOG`

> U **tuned** modelů se očekává, že model už predikuje tyhle 4 třídy (přímo nebo přes mapping).  
> U **pretrained (COCO)** se používá “baseline” mapování (person/bicycle/dog/skis) + jednoduchá heuristika.

---

## Instalace přes `uv`

V kořeni repozitáře (kde je `pyproject.toml`):

```bash
uv venv
# Windows:
# .venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

uv pip install -e .
```

Test, že balíček žije:

```bash
uv run python -c "import counter; print(counter.__name__)"
```

---

## Predikce / počítání

Konfigurace:
- `configs/predict.yaml` – co se má počítat (video, čára, thresholdy, export, preview)
- `configs/models.yaml` – registry modelů + mapping

Spuštění:

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml
```

### Nejdůležitější položky v `configs/predict.yaml`
- `model_id`: klíč z `configs/models.yaml` (např. `rfdetr_pretrained/small`)
- `device`: `cpu` / `cuda:0`
- `videos_dir` + `videos`: vstupní soubory
- `line.coords`: `[x1, y1, x2, y2]` v pixelech
- `line.default_resolution`: základní rozlišení, ve kterém byly coords nakreslené (automaticky se škáluje)
- `greyzone_px`: tolerance okolo čáry (omezuje “kmitání”)
- `thresholds.conf` / `thresholds.iou`
- `tracking.type`: `none` | `bytetrack`
- `export.*`: ukládání výstupů (video/raw/counts)
- `preview.*`: živý náhled

---

## Model registry (`configs/models.yaml`)

Každý model má:
- `backend`: `yolo` nebo `rfdetr`
- `variant`: `tuned` nebo `pretrained`
- `weights`: cesta / název weights
- `mapping`: význam závisí na variantě

### Tuned modely
Mapping je **raw_class_id → kanonická třída** (typicky už 0..3):

```yaml
mapping:
  tourist: 0
  skier: 1
  cyclist: 2
  tourist_dog: 3
```

### Pretrained (COCO) modely
Mapping říká, jaké raw ID odpovídá COCO třídám, které používáme jako baseline:

- `tourist` = **person**
- `cyclist` = **bicycle**
- `tourist_dog` = **dog**
- `skier` = **skis**

Příklad (RF-DETR COCO):
```yaml
mapping:
  tourist: 1      # person
  cyclist: 2      # bicycle
  tourist_dog: 18 # dog
  skier: 35       # skis
```

Heuristika (baseline) typicky dělá:
- `cyclist = bicycle`
- `skier = skis`
- `tourist = max(person - bicycle - skis, 0)`
- `tourist_dog = dog`

> Pokud ti “nesedí třídy”, není to magie: jen máš špatně `mapping` pro daný pretrained model.

---

## Výstupy

Predikce ukládá do `output_dir` (v `configs/predict.yaml`, default `runs/predict`).

Typicky najdeš:
- `*.counts.json` (IN/OUT pro třídy `0..3`)
- `*.pred.mp4` (pokud `export.save_video: true`)
- debug logy (pokud `debug: true`)

---

## Evaluace

Konfigurace v `configs/eval.yaml`:
- `gt_dir`: ground-truth `*.counts.json`
- `runs_dir`: kde jsou predikční runy (např. `runs/predict`)
- `out_dir`: kam uložit eval výstupy
- `filters.*`: výběr backendů / variant / model_ids
- `charts.enabled`: generování grafů

Spuštění:

```bash
uv run python -m counter.eval --config configs/eval.yaml
```

---

## Praktické debug tipy

- Nejdřív si pusť `preview.enabled: true` a dej `preview.every_n_frames` třeba 20 → rychle uvidíš, jestli je čára OK.
- U pretrained modelů vždycky ověř, že `mapping` odpovídá realitě modelu (COCO ID se u různých implementací občas liší v praxi).
- Pokud tracking blbne, zkus dočasně `tracking.type: none` (oddělíš detekci od trackingu).

