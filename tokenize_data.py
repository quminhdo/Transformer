import os
import sys
import nltk
from rakutenma import RakutenMA

class Japanese_Tokenizer:
    def __init__(self):
        rma = RakutenMA()
        rma.load("model_ja.json")
        rma.hash_func = rma.create_hash_func(15)
        self.rma = rma 

    def __call__(self, text_list):
        out = []
        for i in range(len(text_list)):
            out.append(self.tokenize(text_list[i]))
            if (i+1)%1000 == 0:
                print("{} lines tokenized".format(i+1))
        return out
        
    def tokenize(self, text):
        x = self.rma.tokenize(text)
        return [t[0] for t in x]

class English_Tokenizer:
    def __init__(self):
        pass

    def __call__(self, text_list):
        return [self.tokenize(text) for text in text_list]

    def tokenize(self, text):
        return nltk.word_tokenize(text)

def read_file(file_path):
    with open(file_path) as fin:
        return fin.read().splitlines()

def write_tokenized_data_to_file(data, file_path):
    text_list = [' '.join(tokenized_text) for tokenized_text in data]
    with open(file_path, 'w') as fout:
        for text in text_list:
            fout.write(text + "\n")

def process(lines):
    lines = [line.split() for line in lines]
    return [''.join(line) for line in lines]

def tokenize_data(data_path):
#    japanese_tokenizer = Japanese_Tokenizer()
    english_tokenizer = English_Tokenizer()
#    print("Tokenizing Japanese text...")
#    print("reading and processing...")
#    train_ja = process(read_file(os.path.join(data_path, "train.ja")))
#    val_ja = process(read_file(os.path.join(data_path, "val.ja")))
#    test_ja = process(read_file(os.path.join(data_path, "test.ja")))
#    print("tokenizing...")
#    train_ja = japanese_tokenizer(train_ja)
#    val_ja = japanese_tokenizer(val_ja)
#    test_ja = japanese_tokenizer(test_ja)
    print("Tokenizing English text...")
    train_en = english_tokenizer(read_file(os.path.join(data_path, "train.en")))
    val_en = english_tokenizer(read_file(os.path.join(data_path, "val.en")))
    print("Writing tokenized text to new files...")
#    write_tokenized_data_to_file(train_ja, os.path.join(data_path, "train_ja"))
#    write_tokenized_data_to_file(val_ja, os.path.join(data_path, "validate_ja"))
#    write_tokenized_data_to_file(test_ja, os.path.join(data_path, "test_ja"))
    write_tokenized_data_to_file(train_en, os.path.join(data_path, "train_en"))
    write_tokenized_data_to_file(val_en, os.path.join(data_path, "validate_en"))

if __name__ == "__main__":
    assert len(sys.argv) == 2
    tokenize_data(sys.argv[1])
