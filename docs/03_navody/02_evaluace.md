# Jak evaluovat výsledky predikce proti referenční hodnotě

Tento návod popisuje podrobnější ověření systému. Postupujte podle něj tehdy, když chcete porovnat výstupní soubory predikce s referenční hodnotou a získat metriky kvality.

## Předpoklady

Před spuštěním evaluace připravte tyto tři věci:

1. **Konfigurační soubor** `configs/eval.yaml` s cestami ke ground truth a výstupům predikce.
2. **Ground truth** — složka s referenčními soubory ve formátu `<video>.counts.json`.
3. **Výstupy predikce** — složka běhu obsahující `run.json` a soubory `<video>.counts.json`.

### Konfigurační soubor `configs/eval.yaml`

```yaml
# Evaluation configuration
gt_dir: data/counts_gt
runs_dir: runs/to_eval_test/compare_line
out_dir: runs/to_eval_test/compare_line/eval

only_completed: true

videos_dir: ""  # leave empty if you do not want rate-based metrics
rank_by: video_mae_total

filters:
  run_ids: []
  backends: []
  variants: []
  model_ids: []

charts:
  enabled: true
```

Klíčové položky:

| Položka | Popis |
|---------|-------|
| `gt_dir` | Cesta ke složce s ground truth soubory |
| `runs_dir` | Složka s podsložkami jednotlivých predikcí |
| `out_dir` | Cíl pro výstupy evaluace |
| `rank_by` | Metrika pro řazení výsledků (např. `video_mae_total`) |
| `filters` | Omezení na konkrétní běhy, backendy, varianty nebo modely — prázdný seznam znamená žádné omezení |

### Formát ground truth

Pro každé video vytvořte soubor `<video>.counts.json` ve složce `gt_dir`. Příklad `vid16.counts.json`:

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

Číselné klíče odpovídají třídám: 0 = turista, 1 = lyžař, 2 = cyklista, 3 = turista se psem.

### Struktura složky s predikcí

Každý běh predikce musí obsahovat soubor `run.json` s metadaty a soubory `<video>.counts.json` pro každé zpracované video. Příklad `run.json`:

```json
{
  "run_id": "20260316_173022",
  "status": "completed",
  "model_id": "rfdetr_tuned/rfdetr_large",
  "backend": "rfdetr",
  "variant": "tuned",
  "videos": ["vid16.mp4", "vid17.mp4", "vid18.mp4"]
}
```

Evaluace zpracuje pouze běhy se `"status": "completed"`, pokud je v konfiguraci nastaveno `only_completed: true`.

## Spuštění evaluace

Spusťte evaluaci příkazem:

```bash
uv run python -m counter.eval --config configs/eval.yaml
```

Pro evaluaci jednoho konkrétního běhu předejte jeho cestu explicitně:

```bash
uv run python -m counter.eval --config configs/eval.yaml --predict_run_dir runs/predict/<run_dir>
```

Počkejte, až evaluace vytvoří výstupní adresář s touto strukturou:

```
eval_<timestamp>/
├── benchmark.json          # souhrn všech běhů
├── metrics.json            # agregované metriky
├── per_run_metrics.csv     # skóre za každý běh
├── per_video_metrics.csv   # skóre za každé video
├── per_class_metrics.csv   # rozpad po třídách
└── charts/                 # grafy
    ├── leaderboard_score_total_event_wape.png
    ├── heatmap_abs_total_error_IN.png
    ├── heatmap_abs_total_error_OUT.png
    ├── scatter_mae_total_in_vs_out.png
    └── <model>__<video>__<směr>.png  (průběhové grafy per video)
```

## Interpretace výsledků

Klíčovou metrikou je **WAPE** (Weighted Absolute Percentage Error) — vážená průměrná procentuální chyba přes všechna videa. Čím nižší hodnota, tím přesnější model.

1. Otevřete graf `charts/leaderboard_score_total_event_wape.png` pro rychlé porovnání všech modelů podle WAPE.
2. Otevřete `per_run_metrics.csv` a zkontrolujte celkové skóre každého běhu.
3. Otevřete `per_video_metrics.csv` a zjistěte, na kterých videích model chyboval nejvíc.
4. Otevřete `per_class_metrics.csv` a ověřte, zda model neselhává jen na konkrétní třídě objektů.

Porovnejte hodnoty s referenčním ground truth a vyhodnoťte, zda je chyba přijatelná pro zamýšlené použití.

## Návaznost
- Další část: [Shrnutí](../04_zaver/01_shrnuti.md)
