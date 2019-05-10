import tensorflow as tf
import numpy as np
import os
from argparse import ArgumentParser
import models, utils, config

LANG_DICT = {"en": "English", "ja": "Japanese"}

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--data", "-d", help="Path of the directory containing data")
    parser.add_argument("--model", "-m", help="aka model_number. Parameters are saved to or restored from a directory named 'source_language'-'target_language'.'model_number'")
    parser.add_argument("--language", "-l", help="format: 'source_language'-'target_language', use only the first two letters")
    parser.add_argument("--epoch", "-e", type=int, help="checkpoint. default checkpoint is the latest epoch.")
    args = parser.parse_args()
    data = args.data
    model = args.model
    epoch = args.epoch
    languages = {"source":args.language[:2], "target":args.language[-2:]}
    model_path = "{}.{}_{}.{}".format(data, languages["source"], languages["target"], model)
    translate = models.Inference_Model(data, languages, model_path, epoch)
    with open(os.path.join(data, "test.{}".format(languages["source"]))) as f:
        lines = f.read().splitlines()

    with open(os.path.join(model_path, "translation_{}.{}".format(epoch, languages["target"])), 'w') as f:
        for line in lines:
            translated_line, _, tokenized_outputs, _, _ = translate([line])
            tokenized_outputs = np.squeeze(tokenized_outputs)
#            f.write("".join(translated_line) + "\n")
            f.write(" ".join(tokenized_outputs[:-1])+"\n")

