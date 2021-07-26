from keras.layers import Conv2D, multiply


def spatial_attention_block(input):
    '''
    Args:
        input: input tensor

    Returns: a keras tensor

    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x
