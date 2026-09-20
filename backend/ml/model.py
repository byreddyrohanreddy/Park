import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class MKLFusionLayer(layers.Layer):
    """Learnable multi-kernel fusion of a concatenated feature vector."""

    def __init__(self, output_dim=32, n_centers=16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.n_centers = n_centers

    def build(self, input_shape):
        concat_dim = input_shape[-1]
        self.centers = self.add_weight(
            shape=(self.n_centers, concat_dim), initializer="glorot_uniform",
            trainable=True, name="rbf_centers")
        self.gamma = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Constant(1.0 / concat_dim),
            trainable=True, name="rbf_gamma")

        self.linear_proj = layers.Dense(self.output_dim, name="linear_kernel")
        self.poly_proj = layers.Dense(self.output_dim, name="poly_kernel")
        self.rbf_proj = layers.Dense(self.output_dim, name="rbf_kernel")

        self.mix_weights = self.add_weight(
            shape=(3,), initializer="ones", trainable=True, name="kernel_mix_weights")
        super().build(input_shape)

    def call(self, x):
        linear_view = self.linear_proj(x)
        poly_view = self.poly_proj(tf.square(x))

        diffs = tf.expand_dims(x, 1) - tf.expand_dims(self.centers, 0)
        sq_dists = tf.reduce_sum(tf.square(diffs), axis=-1)
        rbf_sims = tf.exp(-self.gamma * sq_dists)
        rbf_view = self.rbf_proj(rbf_sims)

        weights = tf.nn.softmax(self.mix_weights)
        fused = weights[0] * linear_view + weights[1] * poly_view + weights[2] * rbf_view
        return fused

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "n_centers": self.n_centers})
        return config


def build_model(spec_shape=(200, 64), acous_shape=(13,), l1=1e-5, l2=1e-5, dropout_rate=0.3) -> Model:
    """
    Build the True Multi-Modal CNN+RNN+MKL+MLP Model.
    """
    reg = regularizers.l1_l2(l1=l1, l2=l2)

    # 1. Spectrogram Branch
    spec_input = layers.Input(shape=spec_shape, name="spectrogram")
    
    # CNN
    c = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", kernel_regularizer=reg)(spec_input)
    c = layers.BatchNormalization()(c)
    c = layers.MaxPooling1D(pool_size=2)(c)
    
    c = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", kernel_regularizer=reg)(c)
    c = layers.BatchNormalization()(c)
    c = layers.MaxPooling1D(pool_size=2)(c)
    c = layers.Dropout(dropout_rate)(c)

    # RNN
    r = layers.LSTM(64, return_sequences=False, kernel_regularizer=reg)(c)
    r = layers.Dropout(dropout_rate)(r)

    # 2. Acoustic Branch
    acous_input = layers.Input(shape=acous_shape, name="acoustic")
    a = layers.Dense(32, activation="relu", kernel_regularizer=reg)(acous_input)

    # 3. MKL Fusion
    concat = layers.Concatenate()([r, a])
    fused = MKLFusionLayer(output_dim=64)(concat)

    # 4. MLP Output
    m = layers.Dense(32, activation="relu", kernel_regularizer=reg)(fused)
    m = layers.Dropout(dropout_rate)(m)
    output = layers.Dense(2, activation="softmax", name="prediction")(m)

    model = Model(inputs=[spec_input, acous_input], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
