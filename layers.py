import tensorflow as tf
import sublayers
class Embedding_Layer:
    def __init__(self, vocab_size, embedding_dim, name):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = tf.get_variable(name, shape=[vocab_size, embedding_dim]) * tf.sqrt(tf.to_float(embedding_dim))

    def __call__(self, x):
        return tf.gather(self.embedding, x)

    def linear(self, x):
        shape = tf.shape(x)
        batch_size, max_length = shape[0], shape[1]
        x = tf.reshape(x, [batch_size * max_length, self.embedding_dim])
        logits = tf.matmul(x, self.embedding, transpose_b=True)
        return tf.reshape(logits, [batch_size, max_length, self.vocab_size])

class Positional_Encoder:
    def __init__(self):
        pass

    def __call__(self, x):
        max_length = tf.shape(x)[1]
        embedding_dim = tf.shape(x)[2]
        positions = tf.range(tf.to_float(max_length), dtype=tf.float32)
        dim_scale = tf.exp(-tf.log(10000.0)*tf.range(tf.to_float(embedding_dim), delta=2.0)/tf.to_float(embedding_dim)) 
        PE_even = tf.sin(tf.expand_dims(positions, 1)*tf.expand_dims(dim_scale, 0))
        PE_odd = tf.cos(tf.expand_dims(positions, 1)*tf.expand_dims(dim_scale, 0))
        PE = tf.concat([tf.expand_dims(PE_even, 2), tf.expand_dims(PE_odd, 2)], 2)
        PE = tf.reshape(PE, [max_length, embedding_dim])
#        PE = tf.concat([PE_even, PE_odd], 1)
        return x + tf.expand_dims(PE, 0)

class Encoder_Stack:
    def __init__(self, n_layers, n_heads, d_model, attention_dim, hidden_size, disable_layer_norm=False):
        self.layers = []
        for i in range(n_layers):
            with tf.variable_scope("encoder_layer_%d"%i):
                with tf.variable_scope("self_attention"):
                    self_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    self_attention_layer = sublayers.Add_Norm_Wrapper(self_attention_layer, disable_layer_norm)
                with tf.variable_scope("feed_forward"):
                    feed_forward_layer = sublayers.Feed_Forward(hidden_size, d_model)
                    feed_forward_layer = sublayers.Add_Norm_Wrapper(feed_forward_layer, disable_layer_norm)
                layer = [self_attention_layer, feed_forward_layer]
                self.layers.append(layer) 
                
    def __call__(self, x, self_attention_mask):
        for layer in self.layers:
            self_attention_layer = layer[0]
            feed_forward_layer = layer[1]
            x, _ = self_attention_layer(x, vk=x, attention_mask=self_attention_mask)
            x = feed_forward_layer(x)
        return x

class Decoder_Stack:
    def __init__(self, n_layers, n_heads, d_model, attention_dim, hidden_size, disable_layer_norm=False):
        self.layers = []
        for i in range(n_layers):
            with tf.variable_scope("decoder_layer_%d"%i):
                with tf.variable_scope("self_attention"):
                    self_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    self_attention_layer = sublayers.Add_Norm_Wrapper(self_attention_layer, disable_layer_norm)
                with tf.variable_scope("encoder_decoder_attention"):
                    encoder_decoder_attention_layer = sublayers.MultiHead_Attention(n_heads, d_model, attention_dim)
                    encoder_decoder_attention_layer = sublayers.Add_Norm_Wrapper(encoder_decoder_attention_layer, disable_layer_norm)
                with tf.variable_scope("feed_forward"):
                    feed_forward_layer = sublayers.Feed_Forward(hidden_size, d_model)
                    feed_forward_layer = sublayers.Add_Norm_Wrapper(feed_forward_layer, disable_layer_norm)
                layer = [self_attention_layer, encoder_decoder_attention_layer, feed_forward_layer]
                self.layers.append(layer)
    
    def __call__(self, x, self_attention_mask, encoder_outputs, encoder_attention_mask):
        for i, layer in enumerate(self.layers):
#            layer_cache = cache.get("layer_%d"%i) if cache is not None else None
            self_attention_layer = layer[0]
            encoder_decoder_attention_layer = layer[1]
            feed_forward_layer = layer[2]
            x, _ = self_attention_layer(x, vk=x, attention_mask=self_attention_mask)
            x, attention_weights = encoder_decoder_attention_layer(x, vk=encoder_outputs, attention_mask=encoder_attention_mask)
            x = feed_forward_layer(x)
        return x, attention_weights

