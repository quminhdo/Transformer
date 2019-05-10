import tensorflow as tf
import layers

class Transformer:
    def __init__(self, 
        source_vocab_size,
        target_vocab_size,
        d_model,
        n_layers,
        attention_dim,
        n_heads,
        hidden_size,
        go_id,
        eos_id,
        pad_id,
        source_word_vectors=None,
        target_word_vectors=None):
        self.GO = go_id
        self.EOS = eos_id
        self.PAD = pad_id
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.source_embedding_layer = layers.Embedding_Layer(source_vocab_size, d_model, "source_word_embedding", trained_word_vectors=source_word_vectors)
        self.positional_encoder = layers.Positional_Encoder(d_model)
        self.encoder_stack = layers.Encoder_Stack(n_layers, n_heads, d_model, attention_dim, hidden_size)
        self.target_embedding_layer = layers.Embedding_Layer(target_vocab_size, d_model, "target_word_embedding", trained_word_vectors=target_word_vectors)
        self.decoder_stack = layers.Decoder_Stack(n_layers, n_heads, d_model, attention_dim, hidden_size)

    def __call__(self, source_inputs, target_inputs=None, dropout=None, length_penalty_weight=None, coverage_penalty_weight=None, beam_size=None):
        encoder_attention_mask = self.get_encoder_attention_mask(source_inputs)
        encoder_outputs, self_attention_weights = self.encode(source_inputs, encoder_attention_mask, dropout)
        if target_inputs is None:
            outputs, hypotheses, attention_weights, fed_inputs =  self.beam_search_decode(encoder_outputs, encoder_attention_mask, beam_size, length_penalty_weight, coverage_penalty_weight)
            output_dict = {"outputs": outputs, "hypotheses": hypotheses, "attention_weights": attention_weights, "fed_inputs":fed_inputs, "self_attention_weights": self_attention_weights}
        else:
            logits, outputs = self.decode_with_teacher_forcing(target_inputs, encoder_outputs, encoder_attention_mask, dropout)
            output_dict = {"logits": logits, "outputs": outputs}
        return output_dict

    def encode(self, inputs, encoder_attention_mask, dropout):
        embedding_outputs = self.source_embedding_layer(inputs)
        encoder_inputs = self.positional_encoder(embedding_outputs, dropout)
        encoder_outputs, self_attention_weights = self.encoder_stack(encoder_inputs, encoder_attention_mask, dropout)
        return encoder_outputs, self_attention_weights

    def decode_with_teacher_forcing(self, target_inputs, encoder_outputs, encoder_attention_mask, dropout):
        max_decode_length = tf.shape(target_inputs)[1]
        decoder_attention_mask = self.get_decoder_attention_mask(max_decode_length)
        embedding_outputs = self.target_embedding_layer(target_inputs)
        decoder_inputs = self.positional_encoder(embedding_outputs, dropout)
        decoder_outputs, _ = self.decoder_stack(decoder_inputs, decoder_attention_mask, encoder_outputs, encoder_attention_mask, dropout)
        logits = self.target_embedding_layer.linear(decoder_outputs)
        outputs = tf.argmax(tf.nn.softmax(logits), -1, output_type=tf.int32)
        return logits, outputs

    def beam_search_decode(self, encoder_outputs, encoder_attention_mask, beam_size, length_penalty_weight, coverage_penalty_weight, dropout=None, extra_decode_length=50):
        batch_size = tf.shape(encoder_outputs)[0]
        encode_length = tf.shape(encoder_outputs)[1]
        max_decode_length = encode_length + extra_decode_length
        decoder_attention_mask = self.get_decoder_attention_mask(max_decode_length)
        initial_variables = {
            "FED_INPUTS" : tf.fill([batch_size*beam_size, 1], self.GO),
            "ATTENTION_WEIGHTS": tf.zeros([batch_size*beam_size, 1, encode_length]),
            "OUTPUTS" : tf.zeros([batch_size*beam_size, 1], tf.int32),
            "FINISHED" : tf.zeros([batch_size*beam_size], tf.int32),
            "SCORES": tf.zeros([batch_size, beam_size*self.target_vocab_size]),
            "INDEX" : tf.constant(0)
        }
        batch_size_dim = encoder_outputs.shape[0]
        encode_length_dim = encoder_outputs.shape[1]
        variables_shape = {
            "FED_INPUTS" : tf.TensorShape([batch_size_dim*beam_size, None]),
            "ATTENTION_WEIGHTS" : tf.TensorShape([batch_size_dim*beam_size, None, encode_length_dim]),
            "OUTPUTS": tf.TensorShape([batch_size_dim*beam_size, None]),
            "FINISHED": tf.TensorShape([batch_size_dim*beam_size]),
            "SCORES": tf.TensorShape([batch_size_dim, beam_size*self.target_vocab_size]),
            "INDEX": tf.TensorShape([]) 
        }
        encoder_outputs = tf.tile(encoder_outputs, [beam_size, 1, 1])
        encoder_outputs = tf.reshape(encoder_outputs, [beam_size, batch_size, encode_length, self.d_model])
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2, 3])
        encoder_outputs = tf.reshape(encoder_outputs, [batch_size*beam_size, encode_length, self.d_model])

        def gather_nd(params, indices):
            indices_shape = tf.shape(indices)
            indices_size = tf.size(indices)
            i = tf.stack([tf.range(indices_size), tf.reshape(indices, [indices_size])], axis=1)
            p = tf.reshape(params, [indices_size, -1])
            return tf.reshape(tf.gather_nd(p, i), indices_shape)

        def compute_score(i, prev_outputs, logits, sequence_length, attention_weights, fed_inputs):
            cur_softmax = tf.nn.softmax(logits[:, -1:, :])
            cur_probabilities = tf.transpose(cur_softmax, [0, 2, 1])
            def generate_probabilities():
                decoded_length = tf.shape(prev_outputs)[1]
                prev_softmax = tf.nn.softmax(logits[:, :-1, :])
                prev_probabilities = gather_nd(prev_softmax, prev_outputs)
                prev_probabilities = tf.tile(prev_probabilities, [1, self.target_vocab_size])
                prev_probabilities = tf.reshape(prev_probabilities, [batch_size*beam_size, self.target_vocab_size, decoded_length])
                return tf.concat([prev_probabilities, cur_probabilities], -1)
            probabilities = tf.cond(tf.equal(i, 1), lambda: cur_probabilities, lambda: generate_probabilities())
            probabilities = tf.reduce_prod(probabilities, axis=-1) # shape [batch_size*beam_size, target_vocab_size]

            length_penalty = tf.pow((5.0 + tf.cast(sequence_length, tf.float32))/(5.0 + 1.0), length_penalty_weight)
            
            unpadded_pos = tf.cast(tf.not_equal(fed_inputs, self.PAD), tf.float32)
            masked_attention_weights = tf.expand_dims(unpadded_pos, -1)*attention_weights
            coverage_penalty = coverage_penalty_weight*tf.reduce_sum(tf.log(tf.minimum(tf.reduce_sum(masked_attention_weights, 1), 1.0)), -1) # shape [batch_size*beam_size]

            scores = tf.log(probabilities)/tf.expand_dims(length_penalty, -1) + tf.expand_dims(coverage_penalty, -1)
            scores = tf.reshape(scores, [batch_size, beam_size*self.target_vocab_size])
            return scores

        def generate_outputs(i, outputs, scores, _beam_size):
            top_scores, indices = tf.nn.top_k(scores, k=_beam_size) # shape [batch_size, _beam_size]
            top_beams = tf.floordiv(indices, self.target_vocab_size)
            top_beams = _beam_size*tf.expand_dims(tf.range(batch_size), -1)+top_beams
            top_beams = tf.reshape(top_beams, [batch_size*_beam_size])
            top_ids = tf.floormod(indices, self.target_vocab_size)
            top_ids = tf.reshape(top_ids, [batch_size*_beam_size, 1])
            outputs = tf.cond(tf.equal(i, 1), lambda: top_ids, lambda: tf.concat([tf.gather(outputs, top_beams), top_ids], -1))
            return outputs, top_beams

        def continue_decode(fed_inputs, attention_weights, outputs, finished, scores, i):
            return tf.logical_and(tf.less(tf.reduce_sum(finished), tf.size(finished)), tf.less(tf.shape(outputs)[1], max_decode_length))
            
        def step(fed_inputs, attention_weights, outputs, finished, scores, i):
            i += 1
            cur_outputs = outputs[:, -1] # shape [batch_size*beam_size]
            next_ids = (1 - finished) * cur_outputs + finished * self.PAD
            fed_inputs = tf.cond(tf.equal(i, 1), lambda: fed_inputs, lambda: tf.concat([fed_inputs, tf.expand_dims(next_ids, -1)], -1))
            sequence_length = self.get_sequence_length(fed_inputs)
            embedding_outputs = self.target_embedding_layer(fed_inputs)
            decoder_inputs = self.positional_encoder(embedding_outputs, dropout)
            decoder_outputs, attention_weights = self.decoder_stack(decoder_inputs, decoder_attention_mask[:, :, :i, :i], encoder_outputs, encoder_attention_mask, dropout)
            logits = self.target_embedding_layer.linear(decoder_outputs)
            attention_weights = tf.reduce_mean(attention_weights, axis=1)
            scores = compute_score(i, outputs, logits, sequence_length, attention_weights, fed_inputs) 
            outputs, _ = generate_outputs(i, outputs, scores, beam_size)
            cur_outputs = outputs[:, -1]
            finished = tf.maximum(finished, tf.cast(tf.equal(cur_outputs, self.EOS), finished.dtype))
            return fed_inputs, attention_weights, outputs, finished, scores, i
        fed_inputs, attention_weights, outputs, _, scores, i = tf.while_loop(continue_decode, step, list(initial_variables.values()), shape_invariants=list(variables_shape.values()))
        top_outputs, top_beams = generate_outputs(i, outputs[:, :-1], scores, 1)
        attention_weights = tf.gather(attention_weights, tf.reshape(top_beams, [batch_size]))
        return top_outputs, outputs, attention_weights, fed_inputs

    def get_encoder_attention_mask(self, x, neg_inf=-1e15): # shape: [batch_size, max_len]
        padded_positions = tf.cast(tf.equal(x, self.PAD), tf.float32)
        attention_mask = padded_positions * neg_inf
        return tf.expand_dims(tf.expand_dims(attention_mask, 1), 1) # shape: [batch_size, 1, 1, length]

    def get_decoder_attention_mask(self, decode_length, neg_inf=-1e15):
        lower_triangular_matrix = tf.matrix_band_part(tf.ones([decode_length, decode_length]), -1, 0)
        padded_positions = tf.cast(tf.equal(lower_triangular_matrix, self.PAD), tf.float32)
        attention_mask = padded_positions*neg_inf
        return tf.expand_dims(tf.expand_dims(attention_mask, 0), 0) # shape: [1, 1, decode_length, decode_length]

    def get_sequence_length(self, x): # shape [batch_size, max_len]
        unpadded_positions = tf.cast(tf.not_equal(x, self.PAD), tf.int32)
        return tf.reduce_sum(unpadded_positions, -1) # shape [batch_size]

