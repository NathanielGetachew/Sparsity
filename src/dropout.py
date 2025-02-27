import tensorflow as tf

def dropout(x, keep_prob=0.5, training=True):
    """
    Custom dropout implementation using only TensorFlow.
    
    Parameters:
        x (tf.Tensor): Input tensor.
        keep_prob (float): Probability of keeping each unit.
        training (bool): Whether the model is in training mode.
        
    Returns:
        tf.Tensor: Output tensor after applying dropout.
    """
    if not training or keep_prob == 1.0:
        return x

    # Create a binary mask using TensorFlow
    random_tensor = tf.random.uniform(tf.shape(x)) < keep_prob
    random_tensor = tf.cast(random_tensor, dtype=x.dtype)

    # Apply the mask and scale the output
    x = x * random_tensor
    x = x / keep_prob

    return x
