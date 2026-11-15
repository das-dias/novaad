# Installation

novaad needs **Python ≥ 3.10**. It runs on Linux and macOS. On Windows, use [WSL](https://learn.microsoft.com/windows/wsl/install).

## From PyPI

=== "uv"

    ```bash
    uv add novaad
    ```

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install novaad
    ```

This installs the `novaad` Python package and the `novaad` command-line tool:

```bash
novaad --version
```

## From source (development)

The repository uses [uv](https://docs.astral.sh/uv/) and a Makefile. `make dev` creates the virtual environment with every dependency group (tests, notebooks, docs), installs novaad in editable mode, and compiles the protobuf schema in `protos/` into `src/schema/`.

```bash
git clone https://github.com/das-dias/novaad.git
cd novaad
make dev     # needs uv and protoc (brew install protobuf / apt install protobuf-compiler)
make test    # full test suite
```

| Make target | What it does |
|---|---|
| `make dev` | `uv sync --all-groups`, then `make protos` |
| `make test` | runs `pytest` on `test/` and the self-tests in `src/novaad/core.py` |
| `make protos` | compiles `protos/*.proto` into `src/schema/` (`_pb2.py` + `.pyi`) |
| `make docs` | serves this documentation locally at <http://localhost:8000> |

## Tools for the examples

novaad itself only needs the LUTs. The [example notebooks](examples/index.md) also generate those LUTs and verify every design in SPICE, so they need two more tools:

ngspice ≥ 40
:   The circuit simulator used for LUT extraction and verification.

    === "macOS"

        ```bash
        brew install ngspice
        ```

    === "Debian / Ubuntu"

        ```bash
        sudo apt install ngspice
        ```

gf180mcu PDK
:   The open GlobalFoundries 180 nm PDK, installed with [ciel](https://github.com/fossi-foundation/ciel). Notebook 01 installs ciel and the PDK for you; see [LUT extraction](lut-extraction.md#1-install-the-gf180mcu-pdk-with-ciel). By hand:

    ```bash
    pip install ciel
    ciel enable --pdk-family gf180mcu <version>   # version from `ciel ls-remote --pdk-family gf180mcu`
    ```

The notebook dependencies (`pymoo`, `inspice`, `ipykernel`, `nbconvert`) are in the `dev` dependency group, so `make dev` installs them.

## Building the documentation

The docs are built with [Zensical](https://zensical.org), which is in the `docs` dependency group:

```bash
make docs                               # live-reloading server
uv run --group docs zensical build      # static site in ./site
```

Every push to `master` publishes the site to GitHub Pages through `.github/workflows/docs.yml`.
