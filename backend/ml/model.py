"""
Hybrid MLP + CNN + RNN + MKL model for PD voice classification.

Architecture follows the paper's Fig. 7 pipeline:
    Input -> CNN -> RNN -> MKL -> MLP -> Fully Connected -> Output

IMPORTANT NOTE ON THE "MKL" LAYER:
Multiple Kernel Learning is normally an SVM-family technique with no standard
differentiable form, and the paper does not give an exact formula for how it
was implemented inside a neural network. Here it is approximated as a
learnable multi-kernel fusion layer (MKLFusionLayer below), which combines
the CNN and RNN outputs through three kernel-inspired transforms
(linear, polynomial/degree-2, and RBF-similarity-to-learned-centers), with
their combination weights learned jointly with the rest of the network via
backpropagation. This is a practical, honestly-labeled reproduction of the
described architecture, not a literal implementation of classical MKL.
"""

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
        # RBF kernel centers (learnable "landmarks" in feature space)
        self.centers = self.add_weight(
            shape=(self.n_centers, concat_dim), initializer="glorot_uniform",
            trainable=True, name="rbf_centers")
        self.gamma = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Constant(1.0 / concat_dim),
            trainable=True, name="rbf_gamma")

        self.linear_proj = layers.Dense(self.output_dim, name="linear_kernel")
        self.poly_proj = layers.Dense(self.output_dim, name="poly_kernel")
        self.rbf_proj = layers.Dense(self.output_dim, name="rbf_kernel")

        # Learned mixture weights over the three kernel views (softmax-normalized)
        self.mix_weights = self.add_weight(
            shape=(3,), initializer="ones", trainable=True, name="kernel_mix_weights")
        super().build(input_shape)

    def call(self, x):
        linear_view = self.linear_proj(x)
        poly_view = self.poly_proj(tf.square(x))

        diffs = tf.expand_dims(x, 1) - tf.expand_dims(self.centers, 0)   # (B, n_centers, D)
        sq_dists = tf.reduce_sum(tf.square(diffs), axis=-1)              # (B, n_centers)
        rbf_sims = tf.exp(-self.gamma * sq_dists)                        # (B, n_centers)
        rbf_view = self.rbf_proj(rbf_sims)

        weights = tf.nn.softmax(self.mix_weights)
        fused = weights[0] * linear_view + weights[1] * poly_view + weights[2] * rbf_view
        return fused

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "n_centers": self.n_centers})
        return config


def build_model(n_features: int = 26, l1=1e-5, l2=1e-5, dropout_rate=0.35) -> Model:
    """
    Build the MLP+CNN+RNN+MKL hybrid model, compiled and ready to train.

    Pipeline follows the paper's stated Architecture Flow and pseudocode exactly:
        Input -> CNN (produces "feature maps")
               -> feature maps passed to RNN (learns temporal sequences)
               -> MKL takes inputs from BOTH CNN and RNN (a skip connection:
                  CNN's feature maps feed the RNN *and* are also passed
                  directly to the fusion step, alongside the RNN's output)
               -> MLP -> Fully Connected -> Output
    This is sequential with a skip connection, NOT two independent parallel
    branches — see README for why this matters and what an earlier version
    of this file got wrong.
    """
    reg = regularizers.l1_l2(l1=l1, l2=l2)

    inputs = layers.Input(shape=(n_features,), name="acoustic_features")
    x = layers.Reshape((n_features, 1))(inputs)

    # --- CNN: extracts "feature maps" from the raw input ---
    c = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu",
                       kernel_regularizer=reg)(x)
    c = layers.BatchNormalization()(c)
    c = layers.MaxPooling1D(pool_size=2)(c)
    feature_maps = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu",
                                  kernel_regularizer=reg, name="cnn_feature_maps")(c)
    feature_maps = layers.Dropout(dropout_rate)(feature_maps)

    # Pooled summary of the feature maps for the direct CNN->MKL skip connection
    cnn_skip = layers.GlobalAveragePooling1D(name="cnn_skip")(feature_maps)

    # --- RNN: takes the CNN's feature maps as input (sequential, per the paper) ---
    r = layers.Bidirectional(layers.LSTM(
        32, return_sequences=True, kernel_regularizer=reg))(feature_maps)
    r = layers.Dropout(dropout_rate)(r)
    rnn_out = layers.Bidirectional(
        layers.LSTM(16, kernel_regularizer=reg), name="rnn_out")(r)

    # --- MKL: kernel-based fusion of CNN (skip) and RNN outputs ---
    combined = layers.Concatenate()([cnn_skip, rnn_out])
    fused = MKLFusionLayer(output_dim=32, n_centers=16, name="mkl_layer")(combined)
    fused = layers.Dropout(dropout_rate)(fused)

    # --- MLP: higher-level non-linear relationships ---
    m = layers.Dense(32, activation="relu", kernel_regularizer=reg)(fused)
    m = layers.Dropout(dropout_rate)(m)
    m = layers.Dense(16, activation="relu", kernel_regularizer=reg)(m)

    # --- Fully connected + output ---
    out = layers.Dense(2, activation="softmax", name="output")(m)

    model = Model(inputs=inputs, outputs=out, name="MLP_CNN_RNN_MKL")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model