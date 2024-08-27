#!/usr/bin/env bash

# Convert all the mermaid diagrams in ./assets/mermaid.md to png
docker run --rm -u $(id -u):$(id -g) -v ./assets:/data minlag/mermaid-cli -w 8192 -b transparent -i "/data/mermaid.md" --outputFormat "png"
