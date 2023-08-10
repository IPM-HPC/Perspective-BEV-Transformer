import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, df, batch_size=32):
        self.df = df
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.indexes = np.arange(len(self.df))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        bv_coords_x = [tf.convert_to_tensor(self.df.iloc[k]['birdview_coords_x'], dtype=tf.float32) for k in indexes]
        bv_coords_y = [tf.convert_to_tensor(self.df.iloc[k]['birdview_coords_y'], dtype=tf.float32) for k in indexes]
        rgb_coords = [tf.convert_to_tensor(self.df.iloc[k]['front_coords'], dtype=tf.float32) for k in indexes]

        images = []
        for k in indexes:
            cand_img = cv2.imread(self.df.iloc[k]['img_path']) / 255.0
            cand_img = cv2.resize(cand_img, (224, 224)) 
            images.append(cand_img)
        image = tf.convert_to_tensor(images, dtype=tf.float32)

        return [rgb_coords, image], [bv_coords_x, bv_coords_y]
