#!/bin/bash
# Runs node classification on the given graph. Tested on cora and citeseer.
GRAPH="$1"

python train_node_classification.py --graph "KARATE"  --task classify --epochs 300
