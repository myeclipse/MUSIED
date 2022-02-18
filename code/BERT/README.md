# BERT

This code is the implementation for BERT model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers).

## Requirements

- python==3.8.10

- torch==1.7.0

- transformers==4.9.2

## Usage

### Data Preprocess

Put the [processed](../../data) sentence-level results in `data` directory.
```
├── data
│     └── train_sentence.json
│     └── dev_sentence.json
│     └── test_sentence.json
```

### Pre-trained model BERT

Chinese BERT model can be obtained from [here.](https://github.com/huggingface/pytorch-transformers)

Put BERT model into directory `bert_pretrain`.
```
├── bert_pretrain
│     └── config.json
│     └── pytorch_model.bin
│     └── vocab.txt
```

### Train and test the model

Run `run.py` and set paramater `--train` is `True` when train and `False` when test.

