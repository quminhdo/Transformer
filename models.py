import os
import time
import tensorflow as tf
import numpy as np
import utils
import config
from transformer import Transformer

class Training_Model:
    def __init__(self, data_path, languages, model_path, graph):
        self.model_params = config.MODEL_PARAMS
        self.train_params = config.TRAIN_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        if self.base_params["glove"] and languages["source"] == "en":
            self.source_converter = self.get_converter(os.path.join(data_path, "glove_words"), source=True)
        else:
            self.source_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["source"])), source=True)
        self.target_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["target"])), source=False)
        vocab_size = {"source": self.source_converter.size, "target": self.target_converter.size}
        epoch = tf.get_variable(name="epoch", initializer=0, trainable=False)
        self.update_epoch = tf.assign_add(epoch, 1)
        self.learning_rate = self.train_params["learning_rate"] * tf.pow(tf.to_float(self.train_params["decay_rate"]), tf.to_float((epoch - 1) // self.train_params["decay_step"]))
        self.sess = tf.Session(graph=graph)
        # Prepare inputs and target outputs 
        dataset = self.prepare_dataset()
        self.iterator = self.get_iterator(dataset)
        data_batch = self.iterator.get_next()
        source_inputs, target_inputs, self.target_outputs = data_batch[0], data_batch[1], data_batch[2]
        self.target_inputs = target_inputs
        # Build model
        source_word_vectors = None
        if self.base_params["glove"] and languages["source"] == "en":
            source_word_vectors = self.get_source_word_vectors("glove_vectors")
        seq2seq = Transformer(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["d_model"], 
        self.model_params["n_layers"], 
        self.model_params["attention_dim"],
        self.model_params["n_heads"],
        self.model_params["hidden_size"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"],
        source_word_vectors=source_word_vectors
        )
        output_dict = seq2seq(source_inputs, target_inputs, dropout=self.train_params["dropout"])
        logits = output_dict["logits"]
        self.outputs = output_dict["outputs"]
        crossents = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_outputs, logits=logits)
        batch_size = tf.to_float(tf.shape(logits)[0])
        mask_weight = self.get_mask_weight()
        self.loss = tf.reduce_sum(mask_weight*crossents)/batch_size
        self.loss = tf.reduce_sum(crossents)/batch_size
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        # save and restore
        self.saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.tables_initializer())
        self.check_restore_params()

    def __call__(self): 
        self.sess.run(self.iterator.initializer)
        total_loss = 0
        epoch = self.sess.run(self.update_epoch)
        n_batch = 0
        while True:
            try:
                target_inputs, outputs, target_outputs, learning_rate, batch_loss, _ = self.sess.run((self.target_inputs, self.outputs, self.target_outputs, self.learning_rate, self.loss, self.train_op))
                n_batch += 1
                total_loss += batch_loss
                #if n_batch % 10 == 1 and n_batch < 1000:
                #    print("Batch {} loss {}".format(n_batch, batch_loss))
            except tf.errors.OutOfRangeError:
                log = "Epoch {}:\nTime: {}\nTraining loss: {}\nLearning rate: {}\n".format(epoch, time.asctime(time.localtime(time.time())), total_loss/n_batch, learning_rate)
#                print("No of batches:", n_batch)
#                self.check_training(outputs, target_outputs)
                self.saver.save(self.sess, os.path.join(self.model_path, "epoch_%d"%epoch))
                break;
        return log, epoch

    def get_converter(self, vocab_file_path, source):
        return utils.Word_Id_Converter(vocab_file_path, self.UNK, source)

    def get_source_word_vectors(self, file_name):
        vectors = np.loadtxt(os.path.join(self.data_path, file_name))
        return tf.constant(vectors, dtype=tf.float32) 

    def get_mask_weight(self):
        unpadded_pos = tf.cast(tf.not_equal(self.target_outputs, self.PAD), tf.float32) 
        return unpadded_pos

    def check_training(self, output, target, inputs=None):
        if inputs is not None:
            print("Input:")
            print(inputs)
        print("Output:")
        print(output)
        print("Target:")
        print(target)

    def check_restore_params(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing parameters...")
            self.sess.run(tf.global_variables_initializer())
         
    def prepare_dataset(self):
        source_data_file = os.path.join(self.data_path, "train_{}".format(self.languages["source"]))
        target_data_file = os.path.join(self.data_path, "train_{}".format(self.languages["target"]))
        return utils.prepare_dataset(source_data_file, target_data_file, self.source_converter, self.target_converter, self.GO, self.EOS, self.UNK)

    def get_iterator(self, dataset):
        pad_value = tf.cast(self.PAD, tf.int32)
        dataset = dataset.shuffle(4000000, reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(self.train_params["batch_size"], ([None], [None], [None]), (pad_value, pad_value, pad_value), drop_remainder=False)
        return dataset.make_initializable_iterator()

    def update_epoch(self):
        return self.sess.run(tf.assign_add(self.epoch, 1))

class Validating_Model:
    def __init__(self, data_path, languages, model_path, graph):
        self.model_params = config.MODEL_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        if self.base_params["glove"] and languages["source"] == "en":
            self.source_converter = self.get_converter(os.path.join(data_path, "glove_words"), source=True)
        else:
            self.source_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["source"])), source=True)
        self.target_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["target"])), source=False)
        vocab_size = {"source": self.source_converter.size, "target": self.target_converter.size}
        self.sess = tf.Session(graph=graph)
        # Prepare data 
        dataset = self.prepare_dataset()
        self.iterator = self.get_iterator(dataset)
        data_batch = self.iterator.get_next()
        source_inputs, target_inputs, self.target_outputs = data_batch[0], data_batch[1], data_batch[2]
        # Build model
        source_word_vectors = None
        if self.base_params["glove"] and languages["source"] == "en":
            source_word_vectors = self.get_source_word_vectors("glove_vectors")
        seq2seq = Transformer(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["d_model"], 
        self.model_params["n_layers"], 
        self.model_params["attention_dim"],
        self.model_params["n_heads"],
        self.model_params["hidden_size"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"],
        source_word_vectors=source_word_vectors
        )
        output_dict = seq2seq(source_inputs, target_inputs)
        logits = output_dict["logits"]
        crossents = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_outputs, logits=logits)
        batch_size = tf.to_float(tf.shape(logits)[0])
        mask_weight = self.get_mask_weight()
        self.loss = tf.reduce_sum(mask_weight*crossents)/batch_size
        # Save and restore
        self.saver = tf.train.Saver()
        self.sess.run(tf.tables_initializer())

    def __call__(self): 
        self.restore_params()
        self.sess.run(self.iterator.initializer)
        total_loss = 0
        n_batch = 0
        while True:
            try:
                batch_loss = self.sess.run((self.loss))
                total_loss += batch_loss
                n_batch += 1
            except tf.errors.OutOfRangeError:
                log = "Validation loss: {}\n".format(total_loss/n_batch)
                break;
        return log

    def get_converter(self, vocab_file_path, source):
        return utils.Word_Id_Converter(vocab_file_path, self.UNK, source)

    def get_source_word_vectors(self, file_name):
        vectors = np.loadtxt(os.path.join(self.data_path, file_name))
        return tf.constant(vectors, dtype=tf.float32) 

    def get_mask_weight(self):
        unpadded_pos = tf.cast(tf.not_equal(self.target_outputs, self.PAD), tf.float32) 
        return unpadded_pos

    def restore_params(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No saved parameters found")
         
    def prepare_dataset(self):
        source_data_file = os.path.join(self.data_path, "validate_{}".format(self.languages["source"]))
        target_data_file = os.path.join(self.data_path, "validate_{}".format(self.languages["target"]))
        return utils.prepare_dataset(source_data_file, target_data_file, self.source_converter, self.target_converter, self.GO, self.EOS, self.UNK)

    def get_iterator(self, dataset):
        pad_id = tf.cast(self.PAD, tf.int32)
        dataset = dataset.padded_batch(32, ([None], [None], [None]), (pad_id, pad_id, pad_id), drop_remainder=False)
        return dataset.make_initializable_iterator()

class Inference_Model:
    def __init__(self, data_path, languages, model_path, epoch):
        self.model_params = config.MODEL_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        self.tokenizer = self.get_tokenizer()
        self.detokenizer = self.get_detokenizer()
        if self.base_params["glove"] and languages["source"] == "en":
            self.source_converter = self.get_converter(os.path.join(data_path, "glove_words"), source=True)
        else:
            self.source_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["source"])), source=True)
        self.target_converter = self.get_converter(os.path.join(data_path, "vocab_{}".format(languages["target"])), source=False)
        vocab_size = {"source": self.source_converter.size, "target": self.target_converter.size}
        self.sess = tf.Session()
        # Input
        self.input_holder = tf.placeholder(tf.string, [None, None])
        inputs = self.source_converter.word2id(self.input_holder)
        # Build model
        source_word_vectors = None
        if self.base_params["glove"] and languages["source"] == "en":
            source_word_vectors = self.get_source_word_vectors("glove_vectors")
        seq2seq = Transformer(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["d_model"], 
        self.model_params["n_layers"], 
        self.model_params["attention_dim"],
        self.model_params["n_heads"],
        self.model_params["hidden_size"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"],
        source_word_vectors=source_word_vectors
        )
#        output_dict = seq2seq(inputs)
        output_dict = seq2seq(inputs, length_penalty_weight=self.model_params["length_penalty_weight"], coverage_penalty_weight=self.model_params["coverage_penalty_weight"], beam_size=self.model_params["beam_size"])
        fed_inputs = output_dict["fed_inputs"]
        self.attention_weights = output_dict["attention_weights"]
        outputs = output_dict["outputs"]
        hypotheses = output_dict["hypotheses"]
        self.self_attention_weights = output_dict["self_attention_weights"]
        self.outputs = self.target_converter.id2word(outputs)
        self.hypotheses = self.target_converter.id2word(hypotheses)
        self.fed_inputs = self.target_converter.id2word(fed_inputs)
        # Save and restore
        self.saver = tf.train.Saver()
        self.sess.run(tf.tables_initializer())
        self.restore_params(epoch)

    def __call__(self, inputs): 
        tokenized_inputs = self.process_inputs(inputs)
#        print(tokenized_inputs)
        outputs, hypotheses, fed_inputs, attention_weights, self_attention_weights = self.sess.run((self.outputs, self.hypotheses, self.fed_inputs, self.attention_weights, self.self_attention_weights), feed_dict={self.input_holder:tokenized_inputs})
#        self.check_inference(id_inputs, id_outputs)
        outputs, tokenized_outputs = self.process_output(outputs)
        hypotheses, _ = self.process_output(hypotheses)
#        _, tokenized_fed_inputs = self.process_output(fed_inputs)
#        print(tokenized_fed_inputs)
        return outputs, hypotheses, tokenized_inputs, tokenized_outputs, attention_weights, self_attention_weights

    def check_inference(self, id_inputs, id_outputs):
        print("ID inputs")
        print(id_inputs)
        print("ID outputs")
        print(id_outputs)

    def get_tokenizer(self):
        if self.languages["source"] in ["en"]:
            return utils.English_Tokenizer()
        if self.languages["source"] == "ja":
            return utils.Japanese_Tokenizer()

    def get_detokenizer(self):
        if self.languages["target"] in ["en"]:
            return utils.English_Detokenizer()
        if self.languages["target"] == "ja":
            return utils.Japanese_Detokenizer()
        
    def get_converter(self, vocab_file_path, source):
        return utils.Word_Id_Converter(vocab_file_path, self.UNK, source)

    def get_source_word_vectors(self, file_name):
        vectors = np.loadtxt(os.path.join(self.data_path, file_name))
        return tf.constant(vectors, dtype=tf.float32) 

    def process_inputs(self, inputs): # list of text
        tokens_list = self.tokenizer(inputs)
        return tokens_list

    def process_output(self, outputs):
        outputs = outputs.tolist()
        tokenized_outputs = [list(map(lambda byte: byte.decode("utf-8"), seq)) for seq in outputs]
        outputs = self.detokenizer(tokenized_outputs)
        outputs = [output[:output.find("<EOS>")].strip() for output in outputs]
        return outputs, tokenized_outputs # list of text

    def restore_params(self, epoch):
        if epoch:
            checkpoint_path = os.path.join(self.model_path, "epoch_{}".format(epoch))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            checkpoint_path = ckpt.model_checkpoint_path
        if checkpoint_path:
            print("Restoring trained parameters from {}".format(checkpoint_path))
            self.saver.restore(self.sess, checkpoint_path)
        else:
            print("No saved parameters found")
