# Bakalářská práce: Obrazová detekce návštěvníků na Jizerské magistrále

V rámci bakalářské práce obrazová detekce návštěvníků na Jizerské magistrále byl vytvořen systém pro detekci a počítání návštěvníků na základě videí z Jizerské magistrály.

## Rychlý start

### Požadavky

- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- [uv](https://github.com/astral-sh/uv) — správce prostředí pro Python

### Instalace `uv`
#### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Po instalaci `uv` zavřete a znovu otevřete terminál.

#### Spuštění uživatelského rozhraní
```bash
git clone https://github.com/janhorak58/counter.git
cd counter
uv sync
uv run counter-ui
```

### Video návod
Podívejte se na krátký video návod od spuštění rozhraní po kontrolu výsledků.

[![Video návod: Jak spustit webové rozhraní pro provedení predikce na videu](./docs/assets/thumbnail.png)](https://youtu.be/64PHaxcjUcU)

Podrobný postup je popsán v [návodu k webovému rozhraní](./docs/03_navody/webove_rozhrani/index.md).

## Obsah
- [Koncept](./docs/02_koncept/01_cil_reseni.md)
- [Návod: webové rozhraní pro predikci](./docs/03_navody/webove_rozhrani/index.md)
- [Návod: evaluace výsledků](./docs/03_navody/02_evaluace.md)
- [Technická dokumentace](./docs/04_architektura/index.md)

