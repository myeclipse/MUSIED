# BERT

This code is the implementation for BERT model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers).

## Requirements

- python==3.8.10

- torch==1.7.0

- transformers==4.9.2

## Usage

### Data Preprocess

Run `preprocess.py` to obtain the sentence-level input of model. Then put the results into `data` directory.

```
├── data
│     └── train_sentence.json
│     └── dev_sentence.json
│     └── test_sentence.json
```

### Pre-trained model BERT

Chinese BERT model can be obtained from [here.](https://github.com/huggingface/pytorch-transformers)

The result of BERT model is put into directory `bert_pretrain` as follows.
```
├── bert_pretrain
│     └── config.json
│     └── pytorch_model.bin
│     └── vocab.txt
```
