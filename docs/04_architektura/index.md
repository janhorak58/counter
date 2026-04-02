# Technická dokumentace

Tento dokument popisuje interní architekturu systému — jak je kód organizován, jaké moduly existují, jak spolu komunikují a v jakých formátech jsou data ukládána. Je určen vývojářům nebo pokročilým uživatelům, kteří chtějí pochopit systém do hloubky, upravit jeho chování nebo integrovat vlastní komponenty.

---

## Obsah

1. [Přehled architektury](#1-přehled-architektury)
2. [Konfigurace projektu](#2-konfigurace-projektu)
3. [Core — základní typy a I/O](#3-core--základní-typy-a-io)
4. [Predikční pipeline](#4-predikční-pipeline)
5. [Tracking — detekce a sledování](#5-tracking--detekce-a-sledování)
6. [Mapping — mapování tříd](#6-mapping--mapování-tříd)
7. [Counting — logika počítání](#7-counting--logika-počítání)
8. [Evaluační pipeline](#8-evaluační-pipeline)
9. [Webové rozhraní (UI)](#9-webové-rozhraní-ui)
10. [Vstupní a výstupní formáty](#10-vstupní-a-výstupní-formáty)
11. [Adresářová struktura](#11-adresářová-struktura)

---

## 1. Přehled architektury

Systém se skládá ze tří vzájemně propojených celků:

```
┌─────────────────────────────────────────────────────────┐
│                     Webové rozhraní (UI)                 │
│            Streamlit app · job management               │
└────────────────────────┬────────────────────────────────┘
                         │ spouští jako subprocess
          ┌──────────────┴──────────────────┐
          ▼                                 ▼
┌─────────────────────┐         ┌──────────────────────┐
│  Predikční pipeline │         │  Evaluační pipeline  │
│  counter predict    │         │  counter eval        │
└─────────┬───────────┘         └──────────────────────┘
          │
    ┌─────▼───────────────────────────────────────────┐
    │                   Core vrstva                    │
    │  typy · schema · config · IO (video, JSON, CSV) │
    └─────────────────────────────────────────────────┘
```

Predikční pipeline zpracovává videa a produkuje výsledné počty průchodů. Evaluační pipeline tato čísla porovnává s referenčními (ground truth) daty a generuje metriky. Webové rozhraní oba procesy ovládá graficky, bez nutnosti psát příkazy.

---

## 2. Konfigurace projektu

**Soubor:** `pyproject.toml`

### Závislosti

| Oblast | Knihovny |
|---|---|
| Computer vision | `opencv-python`, `pillow` |
| Detekce (YOLO) | `ultralytics>=8.0.0` |
| Detekce (RF-DETR) | `rfdetr==1.4.0.post0` |
| Tracking | `supervision>=0.27.0`, `lap` (Maďarský algoritmus) |
| Deep learning | `torch` |
| Data | `numpy`, `pandas` |
| Konfigurace | `pydantic>=2.0`, `pyyaml` |
| Webové rozhraní | `streamlit`, `streamlit-drawable-canvas` |
| Vizualizace | `plotly` |

> Poznámka: `opencv-python-headless` a `opencv-contrib-python-headless` jsou explicitně zakázány, protože UI vyžaduje plnohodnotné OpenCV s grafickými funkcemi.

### Entry points

```
counter     →  counter.cli:main         # CLI příkazy (predict, eval)
counter-ui  →  counter.ui.__main__:main # Spuštění Streamlit UI
```

---

## 3. Core — základní typy a I/O

**Adresář:** `src/counter/core/`

### 3.1 Datové typy (`types.py`)

```python
BBoxXYXY = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

class LineCoords:
    x1, y1, x2, y2: float  # normalizované souřadnice (0.0–1.0)

class CanonicalClass(IntEnum):
    TOURIST      = 0
    SKIER        = 1
    CYCLIST      = 2
    TOURIST_DOG  = 3

class Side(IntEnum):
    OUT     = -1   # vnější strana linky
    IN      =  1   # vnitřní strana linky
    ON      =  0   # bod leží na lince
    UNKNOWN = 99
```

### 3.2 Pydantic schéma (`schema.py`)

Všechny konfigurační objekty jsou validovány pomocí Pydantic v2.

| Třída | Účel |
|---|---|
| `LineCfg` | Definice počítací linky (název, souřadnice, referenční rozlišení) |
| `ThresholdsCfg` | Prahy detektoru (`conf`, `iou`) |
| `TrackingCfg` | Typ trackeru (`none` / `bytetrack`), cesta k YAML konfigu |
| `ExportCfg` | Co ukládat: video, raw data, JSON s počty |
| `PredictConfig` | Hlavní konfigurace predikce — sdružuje výše uvedené |
| `MappingCfg` | Mapování raw class ID modelu na kanonické třídy |
| `ModelSpecCfg` | Registrace modelu (backend, weights, mapping) |
| `ModelsRegistry` | Slovník všech dostupných modelů načtený z `models.yaml` |
| `EvalConfig` | Konfigurace evaluace (gt_dir, runs_dir, metriky, filtry) |

### 3.3 Loaders (`config.py`)

```python
load_yaml(path)                    → Any
load_pydantic(path, cls)           → T       # YAML → Pydantic model
load_models_registry(path)         → ModelsRegistry
load_predict_config(path)          → PredictConfig
load_eval_config(path)             → EvalConfig
```

### 3.4 I/O moduly (`io/`)

**`video.py`**
```python
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

iter_frames(path) → Iterator[Tuple[int, np.ndarray]]
# Vrací (frame_idx, frame_bgr) pro každý snímek videa
```

**`counts.py`** — čtení výsledných počtů a ground truth dat  
**`json.py`** — atomický zápis JSON souborů s retry logikou  
**`fs.py`** — utility pro práci s adresáři

### 3.5 Pipeline základ (`pipeline/`)

**`base.py`** — `PipelineRunner` orchestruje seznam `Stage` objektů; každá stage dostává `StageContext` se sdíleným stavem a assets.  
**`log.py`** — `JsonlLogger` zapisuje události do `.log.jsonl` souboru; každý záznam je JSON objekt s timestampem.

---

## 4. Predikční pipeline

**Adresář:** `src/counter/predict/`

Hlavní třída `PredictPipeline` (`pipeline.py`) spouští čtyři stages v tomto pořadí:

```
PredictConfig
     │
     ▼
 InitRun ──────────────────── vytvoří adresáře, načte model spec
     │
     ▼
 BuildComponents ──────────── vytvoří provider / mapper / counter / renderer
     │
     ▼
 PredictVideos ─────────────  smyčka přes videa a snímky
     │
     ▼
 FinalizeRun ───────────────  zapíše run.json, agregované počty
```

### Stage: InitRun

- Vytvoří adresářovou strukturu `runs/predict/{model_id}/{timestamp}/`
- Načte `ModelSpecCfg` z registru modelů
- Inicializuje `JsonlLogger`

### Stage: BuildComponents

- Detekuje dostupné zařízení (CUDA / CPU fallback)
- Instanciuje:
  - `TrackProvider` — wrapper kolem detektoru (YOLO nebo RF-DETR)
  - `TrackMapper` — mapování syrových tříd na kanonické
  - `TrackCounter` — logika počítání průchodů
  - `FrameRenderer` — volitelná vizualizace

### Stage: PredictVideos

Hlavní smyčka — pro každé video, pro každý snímek:

```
frame_bgr
    │
    ▼  provider.update(frame)
List[RawTrack]   ← track_id, bbox, score, raw_class_id, raw_class_name
    │
    ▼  mapper.map_tracks(raw_tracks)
List[MappedTrack]  ← přiřazena kanonická třída, nerozpoznané zahozeny
    │
    ▼  counter.update(mapped_tracks)
events: "in" | "out" | "undo_in" | "undo_out"
    │
    ▼  renderer.render(...)   [volitelné]
anotovaný snímek → výstupní video
```

Po zpracování videa: `counter.finalize()` → zápis `*.counts.json`.

### Stage: FinalizeRun

- Zapíše `run.json` s kompletními metadaty (config, váhy, mapper, thresholds)
- Agreguje všechny video-výsledky do `aggregate.counts.json`

---

## 5. Tracking — detekce a sledování

**Adresář:** `src/counter/predict/tracking/`

### Rozhraní `TrackProvider` (`providers.py`)

```python
class TrackProvider(ABC):
    def reset() → None
    def update(frame_bgr: np.ndarray) → List[RawTrack]
    def get_label_map() → Optional[Dict[int, str]]
```

### `UltralyticsYoloTrackProvider` (`yolo_provider.py`)

Využívá `ultralytics` (YOLO11 a starší). Volá `model.track()` pokud je tracking povolen, jinak `model.predict()`. Vrací `RawTrack` pro každý detekovaný/sledovaný objekt.

### `RfDetrTrackProvider` (`rfdetr_provider.py`)

Využívá `rfdetr` knihovnu. Tracking zajišťuje ByteTrack z knihovny `supervision`. Podporované velikosti modelu: `nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge`.

### Typy

```python
@dataclass(frozen=True)
class RawTrack:
    track_id: int
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str
```

---

## 6. Mapping — mapování tříd

**Adresář:** `src/counter/predict/mapping/`

Mapper překládá syrové třídy modelu na kanonické třídy projektu. Konkrétní logika závisí na variantě modelu.

### Rozhraní `MappingPolicy` (`base.py`)

```python
class MappingPolicy(ABC):
    def map_raw(raw_class_id, raw_class_name) → Optional[int]
    def finalize_counts(in_counts, out_counts) → (Dict, Dict)
```

### `TunedMapping` (`tuned.py`)

Používá se pro modely dotrénované na cílové třídě.

- `map_raw()`: přímé mapování `raw_class_id → canonical_id` podle `MappingCfg`
- `finalize_counts()`: počty jsou již kanonické, žádná transformace

### `CocoBaselineMapping` (`pretrained.py`)

Používá se pro základní COCO modely bez dotrénování.

- `map_raw()`: COCO ID → intermediate ID (person=100, bicycle=101, skis=102, dog=103)
- `finalize_counts()`: heuristická konverze na kanonické třídy:

```
tourist     = max(person - bicycle - skis, 0)
skier       = skis
cyclist     = bicycle
tourist_dog = dog
```

### `TrackMapper` (`track_mapper.py`)

Adapter obalující `MappingPolicy`. Volá `policy.map_raw()` pro každý `RawTrack` a zahazuje objekty, jejichž třída nemá mapování.

```python
@dataclass
class MappedTrack:
    track_id: int
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int   # kanonická třída (0–3)
    raw_class_id: int
    raw_class_name: str
    initial_side: Side = Side.UNKNOWN
    current_side: Side = Side.UNKNOWN
```

---

## 7. Counting — logika počítání

**Adresář:** `src/counter/predict/counting/`

### Geometrie (`geometry.py`)

Klasifikace bodu vůči lince pomocí cross-productu:

```python
v = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

v > 0  →  Side.IN
v < 0  →  Side.OUT
v == 0 →  Side.ON
```

Bod objektu je vždy `bottom_center` bounding boxu: `((x1+x2)/2, y2)`.

Souřadnice linky se škálují dle aktuálního rozlišení videa vůči referenčnímu (`line_base_resolution`).

### Stavový automat (`net_state.py`)

Každý sledovaný objekt má vlastní `TrackState`:

```python
class NetState(Enum):
    UNKNOWN       # inicializace
    AT_INITIAL    # na inicializační straně
    AWAY          # na opačné straně

class TrackState:
    initial_side: Side         # strana při prvním setkání
    current_side: Side         # aktuální strana
    net_state: NetState        
    
    voted_class_id: int        # třída dle majority vote
    class_history: Deque[int]  # posledních N snímků
    
    counted_dir: Optional[str] # "in" / "out" / None
    counted_frame: Optional[int]
    
    history: Deque[(x, y)]     # trajektorie
```

Logika `NetStateCounter.update()`:

1. Při prvním setkání: nastav `initial_side`
2. Přechod na opačnou stranu → event `"in"` nebo `"out"`
3. Pokud se objekt vrátí do `oscillation_window_frames` → event `"undo_in"` / `"undo_out"`
4. Třída objektu: `majority vote` z posledních `class_vote_window_frames` snímků

### Hlavní counter (`counter.py`)

```python
class TrackCounter:
    line: LineCoords
    finalize_fn: Callable              # mapper.finalize_counts
    oscillation_window_frames: int = 40
    trajectory_len: int = 40
    class_vote_window_frames: int = 30
    line_base_resolution: Tuple = (1920, 1080)
    
    def reset(video_resolution) → None
    def update(tracks: List[MappedTrack]) → None
    def snapshot_counts() → (Dict[int, int], Dict[int, int])
    def finalize() → (Dict[int, int], Dict[int, int])
```

`finalize()` vrací `(in_counts, out_counts)` jako `{canonical_class_id: count}` po aplikaci `finalize_fn` (mapperova finalizace).

---

## 8. Evaluační pipeline

**Adresář:** `src/counter/eval/`

Hlavní třída `EvalPipeline` (`pipeline.py`) spouští:

```
LoadGTCounts         ← načte ground truth JSONy z gt_dir
DiscoverPredictRuns  ← najde dokončené predikční runy
InitOutput           ← vytvoří výstupní adresáře
EvaluateRuns         ← spočítá metriky pro každý run
RankExportCharts     ← rangování + Plotly grafy + CSV export
```

### Metriky (`eval/logic/metrics.py`)

Pro každé video a třídu se počítá:

| Metrika | Popis |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| bias | Průměrná systematická odchylka |
| within_1pct | Podíl predikci do 1 % od GT |
| within_2pct | Podíl predikci do 2 % od GT |
| macro WAPE | Průměr WAPE přes třídy |
| micro WAPE | Celkový `sum(|err|) / sum(gt)` |

### Objev runů (`eval/logic/discover.py`)

Evaluace automaticky nachází predikční runy prohledáváním `runs_dir`. Metadata jsou čtena z `run.json` v každém runu.

```python
@dataclass
class PredictRunInfo:
    run_id: str
    run_dir: Path
    predict_dir: Path
    model_id: str
    backend: str     # "yolo" / "rfdetr"
    variant: str     # "tuned" / "pretrained"
    status: str
```

### Filtrování runů

`EvalConfig` podporuje filtrování přes `filters`:
```yaml
filters:
  backend: rfdetr
  variant: tuned
  model_id: rfdetr_small
```

---

## 9. Webové rozhraní (UI)

**Adresář:** `src/counter/ui/`

### Spuštění

`counter-ui` spustí Streamlit aplikaci. Vstupním bodem je `ui/__main__.py` → `ui/app.py`.

### Session state (`state.py`)

Streamlit session state je inicializován v `ensure_state_defaults()` s výchozími cestami k souborům a konfiguraci. Klíčové položky:

```python
{
    "ui_project_root": Path,
    "ui_predict_config_path": Path,
    "ui_models_config_path": Path,
    "ui_videos_dir": Path,
    "ui_runs_predict_dir": Path,
    "ui_jobs": {"predict": None, "eval": None},
    ...
}
```

### Views (`views/`)

| Modul | Stránka |
|---|---|
| `predict.py` | Výběr modelu, videa, kreslení linky, spuštění predikce, zobrazení výsledků |
| `browse.py` | Prohlížení výsledků predikce |
| `eval.py` | Spuštění evaluace, rankovací tabulky, grafy metrik |
| `assets.py` | Statické assety |

### Správa úloh (`services/jobs.py`)

Predikce a evaluace běží jako samostatné subprocesy. `JobHandle` sleduje jejich stav:

```python
@dataclass
class JobHandle:
    kind: str
    command: List[str]
    process: subprocess.Popen
    log_queue: queue.Queue[str]
    logs: List[str]
    status: str       # "running" | "completed" | "failed" | "cancelled"
    exit_code: Optional[int]
    output_path: Optional[str]
```

Logy z subprocesu jsou čteny v background threadu a průběžně zobrazovány v UI.

### Ostatní služby (`services/`)

| Modul | Účel |
|---|---|
| `discovery.py` | Objev predikčních a evaluačních runů na disku |
| `configs.py` | Načítání, ukládání a aplikace YAML konfigurací |
| `uploads.py` | Nahrávání videí a modelů do projektu |
| `line_picker.py` | Interaktivní kreslení počítací linky přes snímek videa |

---

## 10. Vstupní a výstupní formáty

### Konfigurační soubory (YAML)

**`configs/models.yaml`** — registr modelů
```yaml
models:
  rfdetr_tuned/rfdetr_small:
    model_id: rfdetr_tuned/rfdetr_small
    backend: rfdetr
    variant: tuned
    weights: models/rfdetr_small.pth
    rfdetr_size: small
    mapping:
      tourist: 1
      skier: 2
      cyclist: 3
      tourist_dog: 4

  yolo_pretrained/yolo11n:
    model_id: yolo_pretrained/yolo11n
    backend: yolo
    variant: pretrained
    weights: yolo11n.pt  # staženo automaticky
```

**`configs/predict_ui.yaml`** — výchozí konfigurace predikce
```yaml
run_id: ""
model_id: rfdetr_tuned/rfdetr_small
device: cpu

videos_dir: data/videos
videos: []

thresholds:
  conf: 0.35
  iou: 0.5

tracking:
  type: bytetrack
  tracker_yaml: null
  params: {}

line:
  name: Line_1
  coords: [846, 404, 1328, 456]
  default_resolution: [1920, 1080]

oscillation_window_frames: 40
trajectory_len: 40
class_vote_window_frames: 30
```

### Výstupní soubory

**`{video_stem}.counts.json`** — výsledné počty pro jedno video
```json
{
  "video": "data/videos/vid1.mp4",
  "line_name": "Line_1",
  "in_count":  {"0": 15, "1": 8, "2": 3, "3": 2},
  "out_count": {"0": 14, "1": 7, "2": 3, "3": 2},
  "meta": {
    "run_id": "20260330_110008",
    "model_id": "rfdetr_small",
    "backend": "rfdetr",
    "variant": "tuned",
    "video": {"fps": 25.0, "frame_count": 10000, "width": 1920, "height": 1080}
  }
}
```

Klíče `in_count` / `out_count` jsou stringová `CanonicalClass` ID:
`"0"` = turista, `"1"` = lyžař, `"2"` = cyklista, `"3"` = turista se psem.

**`run.json`** — metadata predikčního runu
```json
{
  "run_id": "20260330_110008",
  "status": "completed",
  "model_id": "rfdetr_small",
  "backend": "rfdetr",
  "variant": "tuned",
  "weights": "models/rfdetr_small.pth",
  "thresholds": {"conf": 0.35, "iou": 0.5},
  "tracker": {"type": "bytetrack"},
  "line": {"name": "Line_1", "coords": [846, 404, 1328, 456]}
}
```

**`predict.log.jsonl`** — průběžný event log (jeden JSON per řádek)
```jsonl
{"t": "2026-03-30T11:00:08Z", "event": "run_start", "run_id": "..."}
{"t": "2026-03-30T11:00:10Z", "event": "video_start", "video": "vid1.mp4", "fps": 25.0}
{"t": "2026-03-30T11:00:15Z", "event": "count_in", "track_id": 42, "class_id": 0, "frame": 300}
{"t": "2026-03-30T11:00:20Z", "event": "video_end", "in_count": {...}, "out_count": {...}}
```

**`per_run_metrics.csv`** — výstup evaluace (jeden řádek per run)

Sloupce: `run_id`, `model_id`, `backend`, `variant`, `video_mae_total`, `micro_wape_total`, `macro_wape_total`, `rmse_total`, `bias_total`, …

---

## 11. Adresářová struktura

```
counter/
├── pyproject.toml
├── configs/
│   ├── models.yaml               # Registr modelů
│   ├── predict_ui.yaml           # Výchozí konfigurace predikce
│   └── eval.yaml                 # Výchozí konfigurace evaluace
│
├── data/
│   ├── videos/                   # Vstupní videa (.mp4)
│   └── counts_gt/                # Ground truth (.counts.json)
│
├── models/                       # Soubory s vahami modelů
│
├── runs/
│   ├── predict/
│   │   └── {model_id}/
│   │       └── {timestamp}/
│   │           ├── run.json
│   │           └── predict/
│   │               ├── {video}.pred.mp4
│   │               ├── {video}.counts.json
│   │               ├── aggregate.counts.json
│   │               └── predict.log.jsonl
│   └── eval/
│       └── eval_{timestamp}/
│           ├── per_run_metrics.csv
│           ├── per_video_metrics.csv
│           ├── per_class_metrics.csv
│           └── charts/
│
└── src/counter/
    ├── core/                     # Typy, schema, config, I/O
    ├── predict/                  # Predikční pipeline
    │   ├── stages/
    │   ├── tracking/             # YOLO + RF-DETR providery
    │   ├── mapping/              # Mapování tříd
    │   ├── counting/             # Logika počítání průchodů
    │   └── visual/               # Vizualizace (renderer)
    ├── eval/                     # Evaluační pipeline
    │   ├── stages/
    │   └── logic/                # Metriky, ranking, grafy
    └── ui/                       # Streamlit webové rozhraní
        ├── views/
        └── services/
```
