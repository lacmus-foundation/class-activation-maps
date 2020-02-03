#!/usr/bin/env python3.6
# coding: utf-8

import os

from tensorflow.keras import (
    layers as L,
    backend as K,
    callbacks as CB,
)
import tensorflow as tf

import horovod.tensorflow.keras as hvd

import efficientnet.tfkeras as efn

from utils import compose
from dali_utils import AugmentationPipeline, CenterCropPipeline, get_pipeline_outs
from config import DATASET_DIR

os.environ['TF_KERAS'] = '1'
from keras_radam import RAdam  # noqa


hvd.init()
device_id = hvd.local_rank()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(device_id)
K.set_session(tf.Session(config=config))

batch_size = 16 * hvd.size()

backbone = efn.EfficientNetB2(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False,
)

layers = [
    backbone,
    L.Conv2D(256, 3),
    L.BatchNormalization(),
    L.Activation('relu'),
    L.Conv2D(256, 3),
    L.BatchNormalization(),
    L.Activation('relu'),
    L.Conv2D(256, 3),
    L.BatchNormalization(),
    L.Activation('relu'),
    L.Conv2D(1, 1),
    L.Flatten(),
    L.Activation('sigmoid')
]


train_pipeline = AugmentationPipeline(
    root_dir=str(DATASET_DIR / 'Tiles' / 'train'),
    batch_size=batch_size,
    num_threads=2,
    device_id=device_id,
    shard_id=device_id,
    num_shards=hvd.size(),
)

images_tensor, labels_tensor, train_steps = get_pipeline_outs(train_pipeline, device_id)

train_model = compose(
    L.Input(tensor=images_tensor, shape=(224, 224, 3)),
    *layers,
)


# Horovod: adjust learning rate based on number of GPUs.
# opt = O.Adadelta(1.0 * hvd.size())
opt = RAdam()

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)


if hvd.rank() == 0:

    MODEL_NAME = 'EfficientNetB2-NoShift'

    val_pipeline = CenterCropPipeline(
        str(DATASET_DIR / 'Tiles' / 'val'),
        batch_size=16,
        num_threads=12,
        device_id=device_id,
    )
    images_tensor, labels_tensor, val_steps = get_pipeline_outs(val_pipeline, device_id)
    val_model = compose(
        L.Input(tensor=images_tensor, shape=(224, 224, 3)),
        *layers,
    )
    val_model.compile(
        RAdam(),
        loss='binary_crossentropy',
        target_tensors=[labels_tensor],
        metrics=['acc']
    )

    class EvaluateModel(CB.Callback):

        def __init__(self, model, steps):
            super().__init__()
            self.val_model = model
            self.val_steps = steps

        def on_epoch_end(self, epoch, logs={}):
            results = self.val_model.evaluate(steps=self.val_steps, verbose=0)
            for result, name in zip(results, self.val_model.metrics_names):
                logs['val_' + name] = result
            msg = '\r%s' % " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            print()
            print(msg + ' ' * (80 - len(msg)))
            print()

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        EvaluateModel(val_model, val_steps),
        CB.ModelCheckpoint(
            'checkpoints/' + MODEL_NAME + '.weights.{epoch:03d}-{val_loss:.4f}.h5',
            save_weights_only=True
        ),
    ]
else:
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

backbone.trainable = False
train_model.compile(opt, loss='binary_crossentropy',  target_tensors=[labels_tensor])
train_model.fit(
    epochs=2,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    verbose=1 if hvd.rank() == 0 else 0,
)

backbone.trainable = True
train_model.compile(opt, loss='binary_crossentropy',  target_tensors=[labels_tensor])
train_model.fit(
    initial_epoch=2,
    epochs=10,
    steps_per_epoch=train_steps,
    callbacks=callbacks,
    verbose=1 if hvd.rank() == 0 else 0,
)

if hvd.rank() == 0:
    train_model.save_weights(MODEL_NAME + '.weights.h5')
