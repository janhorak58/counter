# Jak evaluovat výsledky proti referenční hodnotě

Tento návod je určený pro podrobnější ověření systému. Postupujte podle něj tehdy, když chcete výstupní soubory porovnat s připravenou referenční hodnotou.

## Předpoklady
Připravte si vstupní video, ground truth ve formátu `counts.json` a zkontrolujte základní konfigurační soubory projektu. Ověřte zejména, že `configs/predict.yaml` ukazuje na správná videa, `configs/models.yaml` obsahuje požadovaný model a `configs/eval.yaml` míří na správnou složku s referenčními daty.

## Spuštění predikce z příkazové řádky
Nejprve upravte soubor `configs/predict.yaml` podle zvoleného videa a požadovaného modelu. Zkontrolujte zejména název modelu, vstupní složku s videi a nastavení čáry pro počítání.

Potom spusťte predikci:

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml
```

Pokud chcete během ladění získat podrobnější výstup, spusťte predikci v debug režimu:

```bash
uv run python -m counter.predict --config configs/predict.yaml --models configs/models.yaml --debug
```

Počkejte na dokončení běhu a poznamenejte si výstupní adresář, do kterého se uložily výsledky.

## Kontrola výstupů predikce
Otevřete výstupní složku běhu a ověřte, že vznikl soubor `counts.json` pro každé zpracované video. Zkontrolujte také soubor `run.json`, průběhové logy a případně anotované video, pokud bylo ukládání videa zapnuto.

Porovnejte seznam zpracovaných videí se vstupem v konfiguraci. Ověřte, že nechybí žádný výsledek a že názvy výstupních souborů odpovídají videím, která chcete následně evaluovat.

![Placeholder: výstupy predikce](../assets/screenshots/06_vystupy_predikce.png)
_Doplnit snímek výstupních souborů predikce._

## Spuštění evaluace
Upravte soubor `configs/eval.yaml` tak, aby odpovídal umístění ground truth a výstupů predikce. Potom spusťte evaluaci:

```bash
uv run python -m counter.eval --config configs/eval.yaml
```

Pokud chcete evaluovat jeden konkrétní běh, předejte jeho cestu explicitně:

```bash
uv run python -m counter.eval --config configs/eval.yaml --predict_run_dir runs/predict/<run_dir>
```

Počkejte, až evaluace vytvoří výstupní adresář s metrikami, CSV přehledy a případnými grafy.

![Placeholder: terminál se spuštěnou evaluací](../assets/screenshots/07_cli_evaluace.png)
_Doplnit snímek spuštění evaluace._

## Interpretace výsledků
Otevřete výsledné soubory evaluace a zaměřte se především na přehledové CSV tabulky. Nejprve zkontrolujte soubor `per_run_metrics.csv`, ve kterém uvidíte hlavní skóre běhu, a potom otevřete `per_video_metrics.csv`, pokud chcete zjistit, na kterých videích model chyboval nejvíc.

Porovnejte hodnoty s referenčním ground truth a vyhodnoťte, zda je chyba přijatelná pro zamýšlené použití. Pokud chcete detailnější pohled, otevřete také rozpad po třídách a ověřte, zda model neselhává jen na konkrétní kategorii objektů.

![Placeholder: přehled metrik evaluace](../assets/screenshots/08_metriky_evaluace.png)
_Doplnit snímek výsledných metrik nebo tabulek._

## Návaznost
- Další část: [Shrnutí](../04_zaver/01_shrnuti.md)
