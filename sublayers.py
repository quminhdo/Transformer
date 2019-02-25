import tensorflow as tf
import utils

class MultiHead_Attention:
    def __init__(self, n_heads, d_model, attention_dim):
        assert attention_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = attention_dim // n_heads
        self.attention_dim = attention_dim
        self.q_dense_layer = tf.layers.Dense(attention_dim, use_bias=False)
        self.k_dense_layer = tf.layers.Dense(attention_dim, use_bias=False)
        self.v_dense_layer = tf.layers.Dense(attention_dim, use_bias=False)
        self.output_dense_layer = tf.layers.Dense(d_model, use_bias=False)

    def __call__(self, q, params):
        vk = params["vk"]
        attention_mask = params["attention_mask"]
        q = self.q_dense_layer(q)
        k = self.k_dense_layer(vk)
        v = self.v_dense_layer(vk)
#        if cache is not None:
#            k = tf.concat([cache.get("k"), k], 1)
#            v = tf.concat([cache.get("v"), v], 1)
#            cache.update({"k":k})
#            cache.update({"v":v})
        q_head, k_head, v_head = [self.split_heads(x) for x in [q, k, v]]
#        attention_mask = tf.expand_dims(attention_mask, 1) # shape [batch_size, 1, length, 1]
        output_head, attention_weights = self.scaled_dot_product_attention(q_head, k_head, v_head, attention_mask)
        output = self.combine_heads(output_head)
        output = self.output_dense_layer(output)
        return output, attention_weights
        
    def split_heads(self, x):
        shape = tf.shape(x)
        batch_size, length = shape[0], shape[1]
        x = tf.reshape(x, [batch_size, length, self.n_heads, self.d_head])
        return tf.transpose(x, [0, 2, 1, 3]) # shape [batch_size, n_heads, length, d_head]

    def scaled_dot_product_attention(self, q, k, v, attention_mask):
        x = tf.matmul(q, k, transpose_b=True) # shape [batch_size, n_heads, q_length, k_length]
        x = x * tf.rsqrt(self.d_head*1.0)
        attention_weights = tf.nn.softmax(x + attention_mask)
        y = tf.matmul(attention_weights, v)
        return y, attention_weights
        
    def combine_heads(self, x):
        shape = tf.shape(x)
        batch_size, length = shape[0], shape[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, length, self.attention_dim])

class Feed_Forward:
    def __init__(self, hidden_size, output_size):
        self.hidden_layer = tf.layers.Dense(hidden_size, activation=tf.nn.relu, use_bias=True)
        self.output_layer = tf.layers.Dense(output_size, use_bias=True)

    def __call__(self, x, params):
        h = self.hidden_layer(x)
        y = self.output_layer(h)
        return y

class Add_Norm_Wrapper:
    def __init__(self, module, disable=False):
        self.module = module
        self.layer_norm = None if disable else Layer_Norm()

    def __call__(self, x, **params):
        y = self.module(x, params)
        if self.layer_norm is None:
            return y
        if isinstance(y, tuple):
            return self.layer_norm(x + y[0]), y[1]
        return self.layer_norm(x + y)

class Layer_Norm:
    def __init__(self):
        self.gain = tf.get_variable(name="norm_gain", initializer=1.0)
        self.bias = tf.get_variable(name="norm_bias", initializer=0.0)
        self.epsilon = 1e-15

    def __call__(self, x):
        mean, var = tf.nn.moments(x, [-1], keep_dims=True)
        norm = (x - mean) * tf.rsqrt(var + self.epsilon)
        norm = self.gain * norm + self.bias
        return norm
        
