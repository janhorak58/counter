# Vstupy a výstupy

Při běhu systému vstupuje do zpracování několik typů dat. Uživatel nejprve vybírá video, nad kterým chce počítání spustit, a dále volí model, jenž bude použit pro detekci objektů. Důležitou součástí vstupu je také čára pro počítání průchodů a konfigurační soubory, které určují parametry běhu.

## Vstupy
- Vstupní video.
- Zvolený model.
- Čára pro počítání průchodů.
- Konfigurační soubory potřebné pro běh.

Výstupem zpracování není jen samotný výsledek počítání. Systém ukládá také pomocné soubory, které usnadňují kontrolu správnosti a pozdější evaluaci. Uživatel pracuje jak s finálními počty, tak s průběhovými logy nebo s anotovaným videem.

## Výstupy
- Počty průchodů ve formátu `counts.json`.
- Logy průběhu zpracování.
- Volitelně anotované video.

## Návaznost
- Předchozí část: [Popis projektu](./02_popis_projektu.md)
- Další část: [Hlavní části systému](./04_hlavni_casti_systemu.md)
