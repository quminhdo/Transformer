import tensorflow as tf
import numpy as np
import os
import socket
import threading
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from argparse import ArgumentParser
import models, utils, config

LANG_DICT = {"en": "English", "ja": "Japanese"}

def change_font(text_list, font):
    for text in text_list:
        text.set_font_properties(font)

def plot_attention_weights(source_words_list, target_words_list, attention_weights):
    font = font_manager.FontProperties(fname="ipag.ttf")
    for i in range(len(source_words_list)):
        source_words = source_words_list[i]
        target_words = target_words_list[i]
        alignment = attention_weights[i]
        fig, ax = plt.subplots()
        im = ax.imshow(alignment)
        ax.set_xticks(np.arange(len(source_words)))
        ax.set_yticks(np.arange(len(target_words)))
        xlabels = ax.set_xticklabels(source_words)
        ylabels = ax.set_yticklabels(target_words)
        change_font(xlabels, font)
        change_font(ylabels, font)
        ax.set_title("Attention Weights")
        fig.tight_layout()
#        plt.show()

def handle_client(clientSocket, addr, translate, languages):
    print("{} is connecting.".format(addr))
    clientSocket.send("{} -> {} Translator\nPress Ctrl-C Enter to terminate.\n".format(LANG_DICT[languages["source"]], LANG_DICT[languages["target"]]).encode())
    while True:
        recv_msg = clientSocket.recv(1024)
        recv_dict = pickle.loads(recv_msg)
        if recv_dict is not None:
            text = recv_dict["text"]
            plot = recv_dict["plot"]
            outputs, hypotheses, tokenized_inputs, tokenized_outputs, attention_weights, self_attention_weights = translate([text])
            sent_dict = {"translation": outputs, "hypotheses": hypotheses}
            if plot:
                sent_dict.update({"tokenized_inputs":tokenized_inputs, "tokenized_outputs": tokenized_outputs, "attention_weights": attention_weights, "self_attention_weights": self_attention_weights})
            clientSocket.sendall(pickle.dumps(sent_dict))
            clientSocket.sendall(b"<EOM>")
        else:
            break;
    clientSocket.close()
    print("{} disconneted.".format(addr))

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--data", "-d", help="Path of the directory containing data")
    parser.add_argument("--model", "-m", help="aka model_number. Parameters are saved to or restored from a directory named 'source_language'-'target_language'.'model_number'")
    parser.add_argument("--language", "-l", help="format: 'source_language'-'target_language', use only the first two letters")
    parser.add_argument("--ip", "-i", help="ip address")
    parser.add_argument("--port", "-p", type=int, help="port number")
    parser.add_argument("--epoch", "-e", type=int, help="checkpoint. default checkpoint is the latest epoch.")
    args = parser.parse_args()
    data = args.data
    model = args.model
    epoch = args.epoch
    languages = {"source":args.language[:2], "target":args.language[-2:]}
    model_path = "{}.{}_{}.{}".format(data, languages["source"], languages["target"], model)
    translate = models.Inference_Model(data, languages, model_path, epoch)
    HOST = args.ip
    PORT = args.port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("listening")
        while True:
            try:
                c, addr = s.accept()
                threading.Thread(target=handle_client, args=(c, addr, translate, languages)).start()
            except KeyboardInterrupt:
                break;
