#!/bin/bash

OUTPUT=./outputs/indexes/bm25
INPUT=./outputs/bm25_collection

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    pixi run python3 -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 20 \
                            -input ${INPUT} \
                            -index ${OUTPUT} \
							-storePositions \
                            -storeDocvectors \
                            -storeRaw
fi
