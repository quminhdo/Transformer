# Transformer
Implementation of [Transformer](https://arxiv.org/pdf/1706.03762.pdf) model using Tensorflow
## 1. Preparation
- A directory containing tokenized training data (e.g. *train_ja*), validation data (e.g. *train_en*) and vocabulary files (e.g. *vocab_ja*, *vocab_en*). I used [RakutenMA](https://pypi.org/project/rakutenma/) to tokenize Japanese text and [nltk](https://www.nltk.org/) to tokenize English text.
## 2. Training
- Modify config.py to change hyperparameters.
- Run `python3 train.py -d *data_directory* -l *source_language-target_language* -d *model_number*` to start training.
- Press Ctrl-c to stop training.
## 3. Translate
- Modify beam_size, coverage_penalty_weight, length_penalty_weight in config.py if necessary.
- Run `python3 translate.py -d *data_directory* -l *source_language-target_language* -d *model_number*`. An image of attention_weights is plotted after each translation.
![attention_example](/images/attention_example.png)
## 4. Other
### 4.1. Socket program
- Run `python3 server.py -d *data_directory* -l *source_language-target_language* -d *model_number* -i *ip_address* -p *port_number*` on a server.
- Put client.py and ipag.ttf file in the same directory. Run `python3 client.py -i *server_ip_address* -p *server_port_number*`. Add *--plot* if you want to plot attention_weights.
