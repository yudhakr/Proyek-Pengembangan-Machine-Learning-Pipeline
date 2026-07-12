import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'label'
TEXT_KEY = 'text'
TITLE_KEY = 'title'


def transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    outputs = {}

    title = tf.strings.strip(inputs[TITLE_KEY])
    text = tf.strings.strip(inputs[TEXT_KEY])
    combined = tf.strings.join([title, text], separator=' ')
    combined = tf.strings.lower(combined)
    combined = tf.strings.regex_replace(combined, '[^a-zA-Z0-9\\s]', ' ')
    combined = tf.strings.regex_replace(combined, '\\s+', ' ')
    combined = tf.strings.strip(combined)
    outputs[transformed_name(TEXT_KEY)] = combined

    label = tf.strings.strip(inputs[LABEL_KEY])
    is_real = tf.cast(tf.equal(label, tf.constant('REAL')), tf.int64)
    outputs[transformed_name(LABEL_KEY)] = is_real

    return outputs
