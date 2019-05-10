import os
import sys
from collections import Counter
import config

def read_file(file_path):
    with open(file_path) as fin:
        return fin.read().splitlines()

def generate_vocabulary(data, threshold=None):
    counter = Counter()
    for tokenized_text in data:
        counter.update(tokenized_text)
    return counter.most_common(threshold)

def make_vocab_file(file_path, vocab_counter):
    with open(file_path, 'w') as fout:
        for word, count in vocab_counter:
            fout.write("{} {}\n".format(word, count))

if __name__ == "__main__":
    assert len(sys.argv) == 2
    data_path = sys.argv[1] 
    param_dict = {"iknow" : config.IKNOW_DATA_PARAMS, "jesc" : config.JESC_DATA_PARAMS, "jesc3" : config.JESC_DATA_PARAMS, "jesc2" : config.JESC_DATA_PARAMS}
    data_params = param_dict[data_path]
    print("Reading tokenized data...")
    train_ja = [line.split() for line in read_file(os.path.join(data_path, "train_ja"))]
    train_en = [line.split() for line in read_file(os.path.join(data_path, "train_en"))]
    print("Generating Japanese vocabulary...")
    vocab_ja = generate_vocabulary(train_ja)
    print("Generating English vocabulary...")
    vocab_en = generate_vocabulary(train_en)
    print("Writing vocabulary to files...")
    make_vocab_file(os.path.join(data_path, "vocab.ja"), vocab_ja)
    make_vocab_file(os.path.join(data_path, "vocab.en"), vocab_en)
