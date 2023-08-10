from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Reshape, Concatenate
from tensorflow.keras.applications import ResNet50
import numpy as np


class PerspectiveTransformer:
    def __init__(self, backbone_model):
        self.backbone_model = backbone_model

    def build(self):
        input_coords = Input(shape=(4,))
        input_shape = Input(shape=(224, 224, 3))

        # --- Input Segment ---
        # > Branch 1
        # Extract feature from perspective view image
        if isinstance(self.backbone_model, ResNet50):
            pv_encoded = Reshape(target_shape=(2048,))(self.backbone_model.layers[175].output)
        else:
            pv_encoded = self.backbone_model.output
            pv_encoded = Reshape(target_shape=(1000,))(pv_encoded)

        # > Branch 2
        # Encode input coordinates
        h = Dense(256, activation='relu')(input_coords)
        h = Dropout(rate=0.25)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(256, activation='relu')(h)

        # Merge feature vectors from pv and coords
        merged = Concatenate()([pv_encoded, h])

        # --- Output Segment ---
        # > Branch 1
        # Decoding into output coordinates (x-axis)
        h = Dense(512, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        output_coords_x = Dense(2)(h)

        # > Branch 2
        # Decoding into output coordinates (y-axis)
        h = Dense(1024, activation='relu')(merged)
        h = Dropout(rate=0.25)(h)
        h = Dense(1024, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(512, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(rate=0.25)(h)
        output_coords_y = Dense(2)(h)

        model = Model(inputs=[input_coords, input_shape], outputs=[output_coords_x, output_coords_y])

        return model

if __name__ == "__main__":
    backbone_model = ResNet50(include_top=True, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
    pt = PerspectiveTransformer(backbone_model)
    model = pt.build()

    dummy_coords = np.random.rand(10, 4)
    dummy_images = np.random.rand(10, 224, 224, 3)
    output_coords_x, output_coords_y = model.predict([dummy_coords, dummy_images])

    print("Output Coordinates X:", output_coords_x)
    print("Output Coordinates Y:", output_coords_y)
