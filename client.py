import numpy as np
import socket
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from argparse import ArgumentParser

def change_font(text_list, font):
    for text in text_list:
        text.set_font_properties(font)

def remove_redundancy(target_words, alignment, symbol):
    try:
        eos_pos = target_words.index(symbol)
        target_words = target_words[:eos_pos]
        alignment = alignment[:eos_pos]
    except ValueError:
        pass
    return target_words, alignment

def plot_attention_weights(source_words_list, target_words_list, attention_weights):
    font = font_manager.FontProperties(fname="ipag.ttf")
    for i in range(len(source_words_list)):
        source_words = source_words_list[i]
        target_words = target_words_list[i]
        alignment = attention_weights[i]
        target_words, alignment = remove_redundancy(target_words, alignment, "<EOS>")
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
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ip", "-i", help="server ip address")
    parser.add_argument("--port", "-p", type=int, help="server port")
    parser.add_argument("--plot", action="store_true", help="plot attention weights")
    args = parser.parse_args()
    HOST = args.ip
    PORT = args.port
    plot = args.plot
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        msg = s.recv(1024)
        print(msg.decode())
        while True:
            try:
                text = input("Input text: ")
                if text == "":
                    print("Empty input. Please try again.")
                    continue
                sent_dict = {"text": text, "plot": plot}
                start = time.time()
                s.sendall(pickle.dumps(sent_dict))
                msg = b""
                while True:
                    packet = s.recv(1024)
                    if packet == b"<EOM>":
                        break;
                    msg += packet
                recv_dict = pickle.loads(msg)
                print("Translation:", ' '.join(recv_dict["translation"]))
                print("Hypotheses:")
                print('\n'.join(recv_dict["hypotheses"]))
                print("Delay in sec:", time.time()-start)
                if plot:
                    plot_attention_weights(recv_dict["tokenized_inputs"], recv_dict["tokenized_inputs"], recv_dict["self_attention_weights"])
                    plot_attention_weights(recv_dict["tokenized_inputs"], recv_dict["tokenized_outputs"], recv_dict["attention_weights"])                    
            except KeyboardInterrupt:
                s.sendall(pickle.dumps(None))
                print("\nThank you for using our translator.")
                break;
