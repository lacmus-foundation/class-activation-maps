from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

import tensorflow as tf

import math


class AugmentationPipeline(Pipeline):
    def __init__(self, root_dir, batch_size, num_threads, device_id,
                 use_shift_scale=False,
                 num_shards=None, shard_id=None):
        super().__init__(batch_size, num_threads, device_id, seed=12)

        self.random_angle = ops.Uniform(range=(0, 360.0))
        self.random = ops.Uniform(range=(0.5, 1.5))
        self.random_coin = ops.CoinFlip()

        self.input = ops.FileReader(file_root=root_dir, random_shuffle=True,
                                    num_shards=num_shards, shard_id=shard_id)

        self.decode = ops.ImageDecoder(device='mixed')
        self.rotate = ops.Rotate(device='gpu', interp_type=types.INTERP_LINEAR)
        self.crop = ops.Crop(device='gpu', crop=(224, 224))
        self.use_shift_scale = use_shift_scale
        if self.use_shift_scale:
            self.shift_scale = ops.RandomResizedCrop(
                device='gpu',
                size=(224, 224),
                interp_type=types.INTERP_LINEAR,
                random_area=(0.3, 1.0),
            )
        self.flip = ops.Flip(device='gpu')
        self.color_twist = ops.ColorTwist(device='gpu')

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.rotate(images, angle=self.random_angle())
        images = self.crop(images)
        if self.use_shift_scale:
            images = self.shift_scale(images)
        images = self.flip(images, horizontal=self.random_coin())
        images = self.color_twist(
            images,
            hue=self.random(),
            saturation=self.random(),
            contrast=self.random(),
            brightness=self.random(),
        )
        return (images, labels.gpu())


class CenterCropPipeline(Pipeline):
    def __init__(self, root_dir, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.FileReader(file_root=root_dir, random_shuffle=True)
        self.decode = ops.ImageDecoder(device='mixed')
        self.crop = ops.Crop(device='gpu', crop=(224, 224))

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.crop(images)
        return (images, labels.gpu())


def get_pipeline_outs(pipe, device_id):
    daliop = dali_tf.DALIIterator()
    pipe.build()  # to get epoch size
    epoch_size = list(pipe.epoch_size().values())[0]
    epoch_size = math.ceil(epoch_size / pipe.batch_size)
    with tf.device("/XLA_GPU:{}".format(device_id)):
        images_tensor, labels_tensor = daliop(
            pipeline=pipe,
            shapes=[(pipe.batch_size, 224, 224, 3), (pipe.batch_size, 1)],
            dtypes=[tf.uint8, tf.int32],
            device_id=device_id,
        )
        images_tensor = tf.cast(images_tensor, tf.float32) / 127.5 - 1.
    return images_tensor, labels_tensor, epoch_size
