# coding: UTF-8
import numpy as np


def load_embedding_dict(path):
    char2vec_file = 'pretrained/char2vec_file.mat.npy'
    word2id_file='pretrained/word2id.npy'
    #
    # if os.path.exists(char2vec_file):
    #     char2vec_mat=np.load(char2vec_file)
    #     word2id=np.load(word2id_file).tolist()
    #     id2word={id:word for id, word in enumerate(word2id)}
    #
    #
    #     return char2vec_mat,word2id,id2word

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    char2vec_mat = []
    word2id=['PAD','UNK']
    id2word={0:'PAD',1:'UNK'}
    char2vec_mat.append(np.random.normal(size=( 300)))
    char2vec_mat.append(np.random.normal(size=(300)))

    lines=lines[1:]
    for id,line in enumerate(lines):
        split = line.split(" ")
        char=split[0]

        id2word[len(word2id)] = char
        word2id.append(char)
        char2vec_mat.append(np.array(list(map(float, split[1:]))))

    char2vec_mat = np.array(char2vec_mat, dtype=np.float32)


    np.save(char2vec_file, char2vec_mat)
    np.save(word2id_file,word2id)

    return char2vec_mat,word2id,id2word

if __name__ == '__main__':
    load_embedding_dict('pretrained/token_vec_300.bin')

