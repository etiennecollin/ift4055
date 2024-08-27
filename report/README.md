# Report

This report for the IFT4055 class at Université de Montréal is written using [Typst](https://github.com/typst/typst).

## Dependencies

- [Typst](https://github.com/typst/typst)
- My custom [typst template](https://github.com/etiennecollin/typst-templates)
  - To compile the document as is, this repository should be cloned in `~/github/` such that the path to the repository is `~/github/typst-templates/...`.
- [Docker](https://docs.docker.com/engine/install/)
  - Only to convert the mermaid diagrams from `./assets/mermaid.md` into png images.

## Usage

Use the following command to open a live preview of the document:

```bash
typst watch --root ~
```

To compile the document into the final PDF, run:

```bash
typst compile --root ~
```

To convert the mermaid diagrams in `./assets/mermaid.md` into png images, run:

```bash
./mermaid2png.sh
```
