import tensorflow as tf
import sublayers

class Embedding_Layer:
    def __init__(self, vocab_size, embedding_dim, name, trained_word_vectors):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        if trained_word_vectors is not None:
            tag_num = 2 if name[:6] == "source" else 4
            tag_embedding = tf.get_variable(name, shape=[tag_num, embedding_dim])
            self.embedding = tf.concat([tag_embedding, trained_word_vectors], axis=0)
        else:
            self.embedding = tf.get_variable(name, shape=[vocab_size, embedding_dim]) * tf.sqrt(tf.cast(embedding_dim, tf.float32))

    def __call__(self, x):
        return tf.gather(self.embedding, x)

    def linear(self, x):
        shape = tf.shape(x)
        batch_size, max_length = shape[0], shape[1]
        x = tf.reshape(x, [batch_size * max_length, self.embedding_dim])
        logits = tf.matmul(x, self.embedding, transpose_b=True)
        return tf.reshape(logits, [batch_size, max_length, self.vocab_size])

class Positional_Encoder:
    def __init__(self, embedding_dim, max_length=1000):
        positions = tf.range(tf.cast(max_length, tf.float32), dtype=tf.float32)
        dim_scale = tf.exp(-tf.log(10000.0)*tf.range(tf.cast(embedding_dim, tf.float32), delta=2.0)/tf.cast(embedding_dim, tf.float32)) 
        PE_even = tf.sin(tf.expand_dims(positions, 1)*tf.expand_dims(dim_scale, 0))
        PE_odd = tf.cos(tf.expand_dims(positions, 1)*tf.expand_dims(dim_scale, 0))
        PE = tf.concat([tf.expand_dims(PE_even, 2), tf.expand_dims(PE_odd, 2)], 2)
        PE = tf.reshape(PE, [max_length, embedding_dim])
        self.PE = tf.expand_dims(PE, 0)

    def __call__(self, x, dropout):
        y = x + self.PE[:, :tf.shape(x)[1], :]
        if dropout:
            y = tf.nn.dropout(y, dropout)
        return y

class Encoder_Stack:
    def __init__(self, n_layers, n_heads, d_model, attention_dim, hidden_size):
        self.layers = []
        for i in range(n_layers):
            with tf.variable_scope("encoder_layer_%d"%i):
                with tf.variable_scope("self_attention"):
                    self_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    self_attention_layer = sublayers.Add_Norm_Wrapper(self_attention_layer)
                with tf.variable_scope("feed_forward"):
                    feed_forward_layer = sublayers.Feed_Forward(hidden_size, d_model)
                    feed_forward_layer = sublayers.Add_Norm_Wrapper(feed_forward_layer)
                layer = [self_attention_layer, feed_forward_layer]
                self.layers.append(layer) 
                
    def __call__(self, x, self_attention_mask, dropout):
        for layer in self.layers:
            self_attention_layer = layer[0]
            feed_forward_layer = layer[1]
            x, self_attention_weights = self_attention_layer(x, vk=x, attention_mask=self_attention_mask, attention_dropout=None, residual_dropout=dropout)
            x = feed_forward_layer(x, feed_forward_dropout=None, residual_dropout=dropout)
            self_attention_weights = tf.reduce_mean(self_attention_weights, axis=1)
        return x, self_attention_weights

class Decoder_Stack:
    def __init__(self, n_layers, n_heads, d_model, attention_dim, hidden_size):
        self.layers = []
        for i in range(n_layers):
            with tf.variable_scope("decoder_layer_%d"%i):
                with tf.variable_scope("self_attention"):
                    self_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    self_attention_layer = sublayers.Add_Norm_Wrapper(self_attention_layer)
                with tf.variable_scope("encoder_decoder_attention"):
                    encoder_decoder_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    encoder_decoder_attention_layer = sublayers.Add_Norm_Wrapper(encoder_decoder_attention_layer)
                with tf.variable_scope("feed_forward"):
                    feed_forward_layer = sublayers.Feed_Forward(hidden_size, d_model)
                    feed_forward_layer = sublayers.Add_Norm_Wrapper(feed_forward_layer)
                layer = [self_attention_layer, encoder_decoder_attention_layer, feed_forward_layer]
                self.layers.append(layer)
    
    def __call__(self, x, self_attention_mask, encoder_outputs, encoder_attention_mask, dropout):
        for i, layer in enumerate(self.layers):
#            layer_cache = cache.get("layer_%d"%i) if cache is not None else None
            self_attention_layer = layer[0]
            encoder_decoder_attention_layer = layer[1]
            feed_forward_layer = layer[2]
            x, _ = self_attention_layer(x, vk=x, attention_mask=self_attention_mask, attention_dropout=None, residual_dropout=dropout)
            x, attention_weights = encoder_decoder_attention_layer(x, vk=encoder_outputs, attention_mask=encoder_attention_mask, attention_dropout=None, residual_dropout=dropout)
            x = feed_forward_layer(x, feed_forward_dropout=None, residual_dropout=dropout)
        return x, attention_weights

