# MR-GCN Classification Test

This repository contains a small artificial and multimodal dataset to test the feature encoders of the MR-GCN in a binary classification setting. The graph connecting these features has been randomly generated, such that all information must come from the features.

## Getting Started

First, generate the input files:

    python mrgcn/mkdataset.py -c test/classification/config.toml -o test/classification/ -v

Next, run the MR-GCN with the generated files as input:

    python mrgcn/run.py -c test/classification/config.toml -i test/classification/CLTEST_<timestamp>.tar -v --dry_run
