# IN_DEPTH_SUMMARY

## 1. Co je tento projekt

`counter` je projekt pro pocitani pruchodu objektu pres zadanou caru ve videu. Prakticky jde o pipeline:

`video -> detekce objektu -> tracking -> urceni strany cary -> zaznam IN/OUT -> export counts.json a volitelne anotovane video`

Projekt kombinuje:

- produkcni/experimentni pipeline pro predikci v `src/counter/predict`
- pipeline pro evaluaci counting vystupu v `src/counter/eval`
- Streamlit UI v `src/counter/ui`
- trenovaci, konverzni a pomocne skripty v `scripts/`

Primarni domena projektu je pocitani pruchodu turistu, lyzaru, cyklistu a turistu se psy ve videich.

## 2. Kanonicke tridy

Napric projektem se pouzivaji fixni canonical class ID:

- `0 = TOURIST`
- `1 = SKIER`
- `2 = CYCLIST`
- `3 = TOURIST_DOG`

To je dulezite pro:

- vystupni `*.counts.json`
- mapovani modelu v `configs/models.yaml`
- ground truth v `data/counts_gt`
- evaluaci v `counter.eval`

## 3. Hlavni koncept systemu

Projekt ma dve hlavni roviny:

### 3.1 Counting pipeline

To je hlavni funkcionalita projektu. U vybraneho videa:

1. zvoli se model z registry
2. model vraci detekce nebo tracky
3. backend provede tracking nebo pouzije existujici tracking
4. raw class ID se premapuji na kanonicke tridy
5. pro kazdy track se sleduje pohyb vzhledem k care
6. pri prechodu z jedne strany na druhou se zaznamena `IN` nebo `OUT`
7. na konci vznikne `*.counts.json` a pripadne `*.pred.mp4`

### 3.2 Evaluacni pipeline

Ta porovnava vystupy counting pipeline s ground truth `*.counts.json`.

Nevyhodnocuje primarne kvalitu bounding boxu, ale kvalitu vysledneho pocitani. To znamena:

- jak presne sedi pocty `IN` a `OUT`
- jak presne sedi pocty po tridach
- jak velka je chyba na urovni videa, behu a tridy
- jak si jednotlive modely stoji v zebricku

## 4. Architektura repozitare

### 4.1 Dulezite adresare

- `src/counter/predict` - predikce a counting pipeline
- `src/counter/eval` - evaluace counting vystupu
- `src/counter/ui` - Streamlit UI
- `src/counter/core` - sdilene schema, IO a utility
- `configs` - YAML konfigurace
- `data/videos` - vstupni videa
- `data/counts_gt` - ground truth pro counting evaluaci
- `models` - lokalni vahy natrenovanych modelu
- `runs/predict` - counting runy
- `runs/eval`, `runs/iter*_eval`, `runs/eval_final` - evaluacni vystupy
- `scripts` - trenovani, konverze datasetu, batch utility, legacy migrace

### 4.2 Dulezite entrypointy

- `python -m counter.predict` - spusti counting pipeline
- `python -m counter.eval` - spusti counting evaluaci
- `counter-ui` nebo `python -m counter.ui` - spusti Streamlit UI
- `eval_model.py` - evaluator detekcnich modelu na datasetu

## 5. Jak funguje predikce

### 5.1 Pipeline

`src/counter/predict/pipeline.py` sklada ctyri stage:

1. `InitRun`
2. `BuildComponents`
3. `PredictVideos`
4. `FinalizeRun`

### 5.2 Co dela kazda stage

#### `InitRun`

- nacte `configs/models.yaml`
- overi, ze `model_id` existuje
- vytvori run directory
- vytvori `predict/`
- zalozi JSONL logger

Aktualni implementace uklada behy podle:

`<export.out_dir>/<model_id>/<timestamp>/`

To je dulezity detail: realna cesta je rizena `export.out_dir`, ne `output_dir`.

#### `BuildComponents`

- vybere backend `yolo` nebo `rfdetr`
- zkontroluje device
- pri nedostupne CUDA fallbackne na CPU
- vytvori provider pro detekci/tracking
- vytvori mapper trid
- vytvori `TrackCounter`
- vytvori renderer vystupniho videa

#### `PredictVideos`

Pro kazde video:

- nacte metadata videa
- prepocita caru z referencniho rozliseni na realne rozliseni videa
- po framech vola provider
- raw tracky premapuje na kanonicke tridy
- posle tracky do counteru
- prubezne uklada logy
- pripadne rendruje anotovane video
- na konci zapise `*.counts.json`

#### `FinalizeRun`

- zapise `run.json`
- zapise `aggregate.counts.json`
- ukonci logicky run

### 5.3 Backendy modelu

#### YOLO

Provider je `UltralyticsYoloTrackProvider`.

Umi:

- `YOLO.track(...)` pri zapnutem trackingu
- `YOLO.predict(...)` pri `tracking.type = none`

Pouziva:

- confidence threshold
- IoU threshold
- volitelne `tracker.bytetrack.yaml`

#### RF-DETR

Provider je `RfDetrTrackProvider`.

Umi:

- tuned modely s lokalnimi checkpointy
- pretrained RF-DETR modely
- tracking pres `supervision.ByteTrack`

Poznamka:

- pretrained RF-DETR muze sahat na interni cache/download mechanismus knihovny
- tuned RF-DETR vyzaduje lokalni checkpoint

### 5.4 Mapovani trid

Projekt podporuje dva rezimy:

#### Tuned modely

Predpoklad je, ze model predikuje nebo lze jednoduse mapovat na 4 kanonicke tridy.

Pouziva se `TunedMapping`:

- raw class ID -> canonical ID `0..3`

#### Pretrained modely

Pouziva se `CocoBaselineMapping`.

Nejde o presne porozumeni scene, ale o baseline heuristiku pres COCO-like bucket tridy:

- `person`
- `bicycle`
- `skis`
- `dog`

Z nich se pak dopocte:

- `cyclist = bicycle`
- `skier = skis`
- `tourist_dog = dog`
- `tourist = max(person - bicycle - skis, 0)`

To znamena, ze pretrained baseline muze fungovat rozumne, ale neni semanticky stejna jako tuned model.

### 5.5 Logika pocitani

Pocitani zajistuje `TrackCounter` a `NetStateCounter`.

Kazdy track:

- ma pocatecni stranu vuci care
- ma aktualni stranu
- pri prechodu na opacnou stranu vygeneruje event
- umi vratit event zpet pri kratke oscilaci

Dulezite mechanismy:

- `greyzone_px` - tolerance v okoli cary proti jitteru
- `oscillation_window_frames` - moznost vratit event zpet pri rychlem navratu
- `class_vote_window_frames` - rolling majority vote tridy tracku
- `trajectory_len` - delka historie trajektorie pro vykresleni

Pouziva se bottom-center bboxu, ne stred boxu.

## 6. Vystupy predikce

Typicky vznikaji:

- `run.json`
- `predict/predict.log.jsonl`
- `predict/<video>.counts.json`
- `predict/<video>.pred.mp4`
- `predict/aggregate.counts.json`

### 6.1 Format `*.counts.json`

Zakladni struktura:

```json
{
  "video": "vid16.mp4",
  "line_name": "Line_1",
  "in_count": {
    "0": 59,
    "1": 0,
    "2": 5,
    "3": 4
  },
  "out_count": {
    "0": 42,
    "1": 0,
    "2": 16,
    "3": 4
  },
  "meta": {
    "variant": "hand-labeled"
  }
}
```

U predikce `meta` navic typicky obsahuje:

- `run_id`
- `model_id`
- `backend`
- `variant`
- `weights`
- `mapping`
- `thresholds`
- `tracker`
- `line`
- `greyzone_px`
- `video`
- `outputs`

## 7. Konfigurace

### 7.1 `configs/predict.yaml`

Ridi counting pipeline.

Klicove polozky:

- `model_id`
- `device`
- `videos_dir`
- `videos`
- `thresholds.conf`
- `thresholds.iou`
- `tracking.type`
- `line.coords`
- `line.default_resolution`
- `greyzone_px`
- `preview.*`
- `export.*`

Prakticky dulezite:

- cara se kresli v referencnim rozliseni a za behu se skaluje na video
- pokud `videos` nechate prazdne, pipeline umi videa discovernout ve `videos_dir`
- skutecny vystupni root ridi `export.out_dir`

### 7.2 `configs/models.yaml`

Je to model registry.

Kazdy zaznam ma typicky:

- `backend`
- `variant`
- `weights`
- `mapping`
- u RF-DETR navic `rfdetr_size`

V aktualnim repozitari jsou zde:

- tuned YOLO modely
- tuned RF-DETR modely
- pretrained YOLO modely
- pretrained RF-DETR modely

### 7.3 `configs/eval.yaml`

Ridi counting evaluaci.

Klicove polozky:

- `gt_dir`
- `runs_dir`
- `out_dir`
- `only_completed`
- `videos_dir`
- `rank_by`
- `filters.*`
- `charts.enabled`

## 8. Streamlit UI

UI je v `src/counter/ui`.

### 8.1 Co umi implementace

V kodu existuji view:

- `predict.py`
- `eval.py`
- `browse.py`
- `assets.py`

Aktualni `src/counter/ui/app.py` ale do defaultni aplikace mountuje pouze prediction view. To znamena:

- predikce je primo dostupna v beznem spuzeni UI
- eval/browse/assets jsou implementovane, ale nejsou soucasti aktualni defaultni landing page

### 8.2 Prediction view umi

- vybrat model z registry
- vybrat video ze slozky
- upravit caru rucne nebo z prvniho framu
- spustit background predikci
- zobrazit progress z JSON logu
- zobrazit posledni vysledek
- prehrat anotovane video
- zobrazit zaznamenane crossing eventy
- zobrazit historii predikci

### 8.3 Eval view umi

- nacist `eval.yaml`
- editovat konfiguraci formularove i pres YAML editor
- validovat YAML
- ulozit konfiguraci
- spustit `counter.eval` jako background job

### 8.4 Browse view umi

- prochazet hotove predict runy
- otevrit `run.json`
- otevrit `*.counts.json`
- prehrat predikovana videa
- prochazet eval vystupy
- zobrazit leaderboard a grafy

### 8.5 Assets view umi

- upload videi
- upload ground truth JSON/ZIP
- upload vah modelu
- auto-registraci modelu do `configs/models.yaml`
- preview assetu
- mazani assetu v povolenych root adresarich

## 9. Navod na spusteni

### 9.1 Lokalni spusteni pres `uv`

Repo obsahuje `pyproject.toml` i `uv.lock`, takze doporucena cesta je `uv`.

#### Instalace zavislosti

```bash
uv sync
```

Pokud chcete editable instalaci podobne jako v README:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### Rychly test importu

```bash
uv run python -c "import counter; print(counter.__name__)"
```

### 9.2 Spusteni UI

```bash
uv run counter-ui
```

Alternativne:

```bash
uv run streamlit run src/counter/ui/app.py
```

Defaultne pobezi na:

`http://localhost:8501`

### 9.3 Spusteni predikce z CLI

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml
```

Pro debug:

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml --debug
```

### 9.4 Docker

Repo ma `Dockerfile` a `docker-compose.yml`.

Spusteni:

```bash
docker compose up --build
```

Na pozadi:

```bash
docker compose up -d --build
```

Zastaveni:

```bash
docker compose down
```

Compose mountuje hlavne:

- `configs`
- `data`
- `models`
- `runs`

## 10. Doporuzeny realny workflow predikce

1. Zkontrolovat `configs/models.yaml`, jestli obsahuje spravny `model_id`.
2. Zkontrolovat `configs/predict.yaml`.
3. Overit `videos_dir` a seznam `videos`.
4. Nastavit caru a referencni rozliseni.
5. Spustit predikci.
6. Otevrit `predict.log.jsonl`, `run.json` a `*.counts.json`.
7. Zkontrolovat, jestli baseline mapping u pretrained modelu dava smysl.

Minimalni priklad pro lokalni data v tomto repozitari:

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml
```

Ve workspace jsou aktualne napr. videa:

- `data/videos/vid16.mp4`
- `data/videos/vid17.mp4`
- `data/videos/vid18.mp4`
- `data/videos/vid_test.mp4`

## 11. Navod na evaluaci counting vystupu

Toto je hlavni evaluace projektu z pohledu pocitani.

### 11.1 Co eval porovnava

`counter.eval` bere:

- predikovane `predict/*.counts.json`
- ground truth `data/counts_gt/*.counts.json`

Pary spojuje podle:

`Path(video).stem`

To znamena, ze `vid16.mp4` v GT se paruje s predikci pro `vid16`.

### 11.2 Zakladni spusteni

```bash
uv run python -m counter.eval --config configs/eval.yaml
```

### 11.3 Evaluace jednoho konkretniho runu

```bash
uv run python -m counter.eval --config configs/eval.yaml --predict_run_dir runs/predict/<run_dir>
```

Nebo primo nad `predict/` podslozkou:

```bash
uv run python -m counter.eval --config configs/eval.yaml --predict_run_dir runs/predict/<run_dir>/predict
```

### 11.4 Co musi byt pripraveno

- `gt_dir` musi obsahovat `*.counts.json`
- `runs_dir` musi obsahovat validni predict runy
- volitelne `videos_dir` muze pomoct s rate-based metrikami

V tomto repozitari existuji GT soubory napr.:

- `data/counts_gt/vid16.counts.json`
- `data/counts_gt/vid17.counts.json`
- `data/counts_gt/vid18.counts.json`
- `data/counts_gt/vid_test.counts.json`

### 11.5 Hlavni vystupy evaluace

Eval vytvori adresar:

`<out_dir>/eval_<timestamp>/`

Typicky obsahuje:

- `eval.log.jsonl`
- `benchmark.json`
- `metrics.json`
- `per_run_metrics.csv`
- `per_video_metrics.csv`
- `per_class_metrics.csv`
- `charts/`

### 11.6 Co znamenaji hlavni metriky

#### `score_total_video_mae`

Prumerna chyba na urovni videa nad class-aware total error pro `IN` a `OUT`.

#### `score_total_micro_wape`

Class-aware micro WAPE:

- chyba se pocita pres tridy
- penalizuje i zameny trid

To je casto nejrozumnejsi headline score pro counting kvalitu.

#### `score_total_macro_wape`

Prumer pres tridy, vhodny kdyz chcete sledovat balans mezi tridami.

#### `score_total_rate_mae`

Chyba po prepoctu na pasaze za hodinu. Funguje jen kdyz jsou dostupne delky videi.

### 11.7 Jak cist CSV vystupy

#### `per_run_metrics.csv`

Leaderboard modelu/runu.

Obsahuje:

- rank
- `run_id`
- `model_id`
- `backend`
- `variant`
- hlavni score
- total MAE/RMSE/WAPE
- volitelne rate metriky

#### `per_video_metrics.csv`

Rozpad po jednotlivych videich.

Vhodne pro zjisteni:

- ktere video dela modelu problem
- zda je chyba hlavne v `IN`, `OUT` nebo konkretni tride

#### `per_class_metrics.csv`

Rozpad po tridach.

Vhodne pro zjisteni:

- jestli model selhava hlavne na `cyclist`
- jestli pretrained baseline podhodnocuje `tourist`
- jestli se model ztraci pri tridnich zamenach

### 11.8 Prakticky doporuceny evaluacni postup

1. Spustit predikci nad stejnou sadou videi jako GT.
2. Zkontrolovat, ze pro kazde GT video existuje `*.counts.json`.
3. Spustit `counter.eval`.
4. Otevrit `per_run_metrics.csv`.
5. Seradit podle `score_total_micro_wape` nebo `score_total_video_mae`.
6. U nejlepsich i nejhorsich modelu otevrit `per_video_metrics.csv`.
7. Podivat se do `charts/` a do `predict.log.jsonl`, kde je potreba.

## 12. Dalsi typ evaluace: evaluace detekcnich modelu

Vedle counting evaluace zde existuje i evaluace detekcnich modelu na datasetu.

To je oddelena vrstva od `counter.eval`.

Pouziva se:

- `eval_model.py`
- `scripts/eval_models.py`

Smysl:

- zmerit mAP, precision, recall, F1 pro detekcni model
- porovnat YOLO a RF-DETR z pohledu detekce
- odlisit problem detekce od problemu counting logiky

### 12.1 Spusteni batch evaluace modelu

```bash
python scripts/eval_models.py --all
```

Nebo napr.:

```bash
python scripts/eval_models.py --yolo-only
python scripts/eval_models.py --rfdetr-only
python scripts/eval_models.py --split test
python scripts/eval_models.py --dry-run
```

Tato cast je dulezita hlavne pro vyvoj modelu, ne pro finalni counting benchmark.

## 13. Training a priprava datasetu

### 13.1 YOLO training

Skript:

`scripts/train_yolo.py`

Umi:

- YOLOv8 a YOLO11 varianty
- resume
- tuning batch/epochs/imgsz
- rizeni augmentaci
- auto-generovani run name

Typicke spusteni:

```bash
python scripts/train_yolo.py --model yolo11l --epochs 100
```

### 13.2 RF-DETR training

Skript:

`scripts/train_rfdetr.py`

Umi:

- varianty `nano/small/base/medium/large`
- scratch workflow
- silne augmentace
- resume
- tuning batch/grad accumulation/lr

Typicke spusteni:

```bash
python scripts/train_rfdetr.py --model medium --epochs 100
```

### 13.3 Konverze YOLO datasetu do COCO

Skript:

`scripts/convert_yolo_to_coco.py`

Pouziti:

```bash
python scripts/convert_yolo_to_coco.py
```

To je potreba hlavne pro RF-DETR training.

## 14. Pomocne a legacy utility

V repozitari jsou i utility pro prechod mezi starsimi a novejsimi formaty.

Dulezite:

- `scripts/legacy_to_counts_json.py` - prevod starsich CSV predikci do noveho `counts.json` formatu
- `scripts/fix_counts_from_raw.py` - oprava `counts.json` z raw bucket poctu
- `scripts/choose_line.py` - rucni vyber cary na prvnim framu videa
- `scripts/generate_eval_visuals.py`, `scripts/generate_eval_matplotlib.py` - generovani evaluacnich vizualizaci
- `scripts/visualize_coco_samples.py` - preview COCO datasetu

To vysvetluje, proc jsou v repozitari videt i starsi struktury runu a starsi formaty metadat.

## 15. Dulezite repository-specific poznamky a limity

### 15.1 `output_dir` vs `export.out_dir`

Dokumentace a schema zminuji `output_dir`, ale aktualni prediction pipeline pouziva pro realny run root `export.out_dir`.

Prakticky:

- chcete-li zmenit misto ukladani behu, sledujte hlavne `export.out_dir`

### 15.2 `run_id` z configu neni hlavni identifikator adresare

Filesystem naming je aktualne timestamp-based.

### 15.3 `probe_frames` je v CLI/schema, ale v hlavni predikcni stage se nepouziva

Je tedy lepsi nespolihat na nej jako na funkcni frame limiter.

### 15.4 UI vs kod

V kodu existuji eval/browse/assets view, ale aktualni `app.py` renderuje jen prediction view.

### 15.5 Repo obsahuje legacy i nove struktury

V `runs/`, `models/` a skriptech jsou videt starsi i novejsi workflow. Napr.:

- starejsi runy typu `runs/predict/rfdetr_tuned__rfdetr_large_v3`
- novejsi pipeline tvari vystup jako `<export.out_dir>/<model_id>/<timestamp>`

Obe vrstvy je potreba pri orientaci v repo rozlisovat.

### 15.6 Datove cesty nejsou vsude konzistentni

Cast skriptu odkazuje na:

- `data/yolo_dataset`
- `data/coco_dataset`

ale aktualni workspace obsahuje i:

- `data/dataset_yolo`
- `data/dataset_coco`

Pri trenovani nebo konverzi je proto nutne nejdriv overit, ktera varianta je pro dany skript platna.

## 16. Kratke doporuceni pro dalsi praci

Pokud chcete projekt pouzivat prakticky a ne jen studovat kod, drzte se tohoto poradi:

1. `uv sync`
2. zkontrolovat `configs/models.yaml`
3. zkontrolovat `configs/predict.yaml`
4. spustit `counter-ui` nebo `counter.predict`
5. zkontrolovat `runs/predict/.../predict/*.counts.json`
6. spustit `counter.eval`
7. porovnat modely podle `per_run_metrics.csv`

To je nejkratsi stabilni cesta, jak se v projektu neztratit mezi counting pipeline, detector training workflow a legacy skripty.
