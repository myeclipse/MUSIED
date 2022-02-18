# BiLSTM

This code is the implementation for `BiLSTM` and `BiLSTM-CRF` model.

## Requirements

--python==3.8.10

--torch==1.7.0

## Usage

### Download character embedding

The Chinese character embeddings `token_vec_300.bin` can be obtain from [here.](https://github.com/liuhuanyong/ChineseEmbedding) Download and put `token_vec_300.bin` in directory `pretrained`.

Run `pre_embedding.py` to obtain `char2vec_file.mat.npy` and `word2id.npy` and put them in directory `pretrained` for the following training.


```
├── pretrained
│     └── token_vec_300.bin
│     └── char2vec_file.mat.npy
│     └── word2id.npy
```

### Data preprocess

Put the [processed](../../data) sentence-level results in `data` directory.

```
├── data
│     └── train_sentence.json
│     └── dev_sentence.json
│     └── test_sentence.json
```

### Train and test

Run `run.py` and set paramater `--train` is `True` when train and `False` when test.
