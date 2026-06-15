import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'label'
TEXT_KEY = 'text'

def transformed_name(key):
    return key + '_xf'

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=32):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn
    )

    def split_label(features):
        text = features[transformed_name(TEXT_KEY)]
        label = features[transformed_name(LABEL_KEY)]
        return text, label

    dataset = dataset.map(split_label)
    return dataset

def build_model(vocab_size=15000, embedding_dim=128, max_len=200):
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=transformed_name(TEXT_KEY))

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_len,
        name='text_vectorizer'
    )

    x = vectorize_layer(inputs)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    model = build_model()

    for texts, _ in train_dataset.take(1):
        model.get_layer('text_vectorizer').adapt(texts)

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
    )

    model.save(fn_args.serving_model_dir, save_format='tf')
