# Spuštění predikce

Na stránce najděte sekci pro výběr modelu a videa.

![Otevření sekce pro výběr modelu a videa](../../assets/screenshots/predict/01_section.png)

Nastavte tyto položky:

- **Varianta modelu** — Pro vlastní natrénovaný model zvolte `tuned`. Pro předtrénovaný model zvolte `pretrained`.
- **Model** — Zvolte model pro predikci.
- **Video** — Zvolte video ke zpracování. Prázdný seznam znamená, že musíte nejprve nahrát video — postupujte podle [návodu k nahrání videa](./01_nahrani_videa.md).
- **Vykreslit každý n-tý snímek** — Určuje jemnost výsledného videa. Výchozí hodnota 5 vykreslí každý pátý snímek.
- **Vlastní čára** — Nastavte vlastní polohu čáry pro počítání průchodů.

## Nastavení vlastní čáry

Klikněte na tlačítko "Vlastní čára" pro rozbalení nabídky.

*Poznámka: před načtením prvního snímku videa proveďte znovunačtení stránky (`CTRL + R`).*

Zvolte jeden ze dvou způsobů nastavení:

1. **Zadejte souřadnice ručně** — Zadejte přesné souřadnice začátku a konce čáry ve formátu `x,y` (např. `100,200`) a vyplňte rozlišení videa.
2. **Nastavte čáru nad prvním snímkem videa (doporučeno)** — Klikněte na tlačítko "Nastavit čáru nad prvním snímkem videa". Načte se první snímek zvoleného videa a zobrazí se nástroj pro nastavení čáry.

![Otevření nastavení pro vlastní čáru](../../assets/screenshots/predict/02_pick_line_1.png)

Prvním kliknutím označte místo, kde má čára začínat. Druhým kliknutím označte místo, kde má čára končit.

![Nastavení vlastní čáry nad prvním snímkem videa](../../assets/screenshots/predict/03_pick_line_2.png)

Klikněte na tlačítko "Použít naklikané body" pro potvrzení nastavení čáry.

## Spuštění

Po nastavení všech parametrů klikněte na tlačítko "Spustit predikci".

![Spuštění predikce](../../assets/screenshots/predict/04_run.png)

## Průběh predikce

Po spuštění predikce se zobrazí průběh běhu.

![Průběh predikce](../../assets/screenshots/predict/05_running.png)

V horní části se zobrazuje aktuální stav běhu, počet zpracovaných snímků, stav, zařízení, zpracovávané video a uplynulý čas. Pro kontrolu průběhu zobrazte detailní logy z běhu.

![Průběh predikce — detail logů](../../assets/screenshots/predict/06_logs_progress.png)

Po dokončení běhu se stav změní na "dokončeno".

![Dokončení predikce](../../assets/screenshots/predict/07_done.png)

- [Předchozí část: Nahrání modelu](./02_nahrani_modelu.md)
- [Další část: Kontrola výsledků predikce](./04_kontrola_vysledku.md)
- [Zpět na přehled návodu](../index.md)
