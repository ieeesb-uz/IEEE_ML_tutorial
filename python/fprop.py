import numpy as np

def fprop(input_batch=None, word_embedding_weights=None,
          embed_to_hid_weights=None, hid_to_output_weights=None,
          hid_bias=None, output_bias=None):
    numwords, batchsize = input_batch.shape
    vocab_size, numhid1 = word_embedding_weights.shape
    numhid2 = embed_to_hid_weights.shape[1]

    # % COMPUTE STATE OF WORD EMBEDDING LAYER.
    # Look up the inputs word indices in the word_embedding_weights matrix.
    foo = word_embedding_weights[np.ndarray.flatten(input_batch - 1), :].T

    embedding_layer_state = np.reshape(foo, (numhid1 * numwords, -1))

    # % COMPUTE STATE OF HIDDEN LAYER.
    # Compute inputs to hidden units.

    inputs_to_hidden_units = np.matmul(embed_to_hid_weights.T, embedding_layer_state) + np.tile(hid_bias,
                                                                                                (1, batchsize))

    # Apply logistic activation function.
    hidden_layer_state = np.divide(1, (1 + np.exp(-inputs_to_hidden_units)))

    # % COMPUTE STATE OF OUTPUT LAYER.
    # Compute inputs to softmax.
    inputs_to_softmax = np.matmul(hid_to_output_weights.T, hidden_layer_state) + np.tile(output_bias, (1, batchsize))
    # Subtract maximum.
    # Remember that adding or subtracting the same constant from each input to a
    # softmax unit does not affect the outputs. Here we are subtracting maximum to
    # make all inputs <= 0. This prevents overflows when computing their
    # exponents.
    inputs_to_softmax = inputs_to_softmax - np.tile(np.max(inputs_to_softmax), (vocab_size, 1))

    # Compute exp.
    output_layer_state = np.exp(inputs_to_softmax)

    # Normalize to get probability distribution.
    output_layer_state = np.divide(output_layer_state, np.tile(np.sum(output_layer_state, 0), (vocab_size, 1)))
    return embedding_layer_state, hidden_layer_state, output_layer_state