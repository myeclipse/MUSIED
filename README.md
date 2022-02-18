# MSInformalED
Dataset and baselines for paper "MSInformalED: A Benchmark for Multi-Source Heterogeneous Informal Text Event Detection".

## Data

The dataset can be obtained from the “data” folder. The data format is introduced in [this document](data/README.md).

### Data preprocess

Run `preprocess.py` to obtain the sentence-level input of model. The result is saved in `data` directory.
```
├── data
│     └── train_sentence.json
│     └── dev_sentence.json
│     └── test_sentence.json
```

## Codes

We release the source codes for the baselines, including 

sentence-level models: 

--[DMCNN](code/DMCNN)

--[BiLSTM](code/BiLSTM)

--[BERT](code/BERT)

--[C-BiLSTM](code/C-BiLSTM)

--[DMBERT](code/DMBERT)

document-level models

--[HBTNGMA](code/HBTNGMA)

--[MLBiNet](code/MLBiNet).
