
# Nahrání modelu

*V aplikaci jsou již k dispozici předtrénované modely. Pro použití předtrénovaného modelu tuto sekci přeskočte a přejděte na [Spuštění predikce](./03_predikce.md). Tato sekce popisuje nahrání vlastního (natrénovaného) modelu.*

---

Nahrajte vlastní model podle těchto kroků:

1. Připravte si model ve formátu PT nebo PTH kompatibilní s architekturou Ultralytics YOLO nebo Roboflow RF-DETR.

2. Rozklikněte rozbalovací menu "Nahrát vlastní váhy modelu".

![Otevření rozbalovacího menu nahrání modelu](../../assets/screenshots/model_upload/01_toggle.png)

3. Vyplňte tato pole:

   - **Backend** — YOLO nebo RF-DETR. Zvolte ten, který odpovídá vašemu modelu.
   - **Varianta** — Výchozí je tuned. Pro model předtrénovaný na obecné datové sadě (např. COCO) zvolte pretrained.
   - **ID modelu** — Identifikátor modelu v systému. Používejte čísla a malá písmena bez diakritiky, mezer a speciálních znaků.
   - **Mapování tříd** — Zadejte čísla tříd z vašeho modelu odpovídající třídám tourist, skier, cyclist, tourist_dog. Podrobnosti jsou na konci této stránky.
   - **Velikost RF-DETR modelu** — Pouze pro RF-DETR. Zvolte velikost modelu nutnou pro správné zpracování.

4. Klikněte na tlačítko "Drag and drop file here" a vyberte soubor s modelem.

![Nahrání modelu](../../assets/screenshots/model_upload/02_submit.png)

5. Klikněte na tlačítko "Nahrát a registrovat".

Po úspěšném nahrání se zobrazí potvrzující hláška. Model je připraven k použití.

![Hláška o úspěšném nahrání modelu](../../assets/screenshots/model_upload/03_done.png)

- [Předchozí část: Nahrání videa](./01_nahrani_videa.md)
- [Další část: Spuštění predikce](./03_predikce.md)
- [Zpět na přehled návodu](../index.md)




## Podrobnější vysvětlení mapování tříd

Pokud váš model rozezná lyžaře jako třídu `0`, cyklistu jako třídu `1` a turistu jako třídu `2`, zadejte tato čísla přesně tak, jak jsou ve vašem modelu. Tato informace slouží softwaru ke správnému určení třídy.

| Třída v systému | Odpovídající třída v modelu |
|-----------------|-----------------------------|
| tourist         | 2                           |
| skier           | 0                           |
| cyclist         | 1                           |

**V datasetu COCO jsou třídy pojmenované následovně:**
| Třída v systému | Odpovídající třída v COCO |
|-----------------|-----------------------------|
| tourist         | person (1)                  |
| skier           | skis (31)                   |
| cyclist         | bicycle (2)                 |
| tourist_dog     | dog (17)                    |
