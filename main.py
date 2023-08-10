import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from utils import read_hdf5_file, evaluate_metrics
from data import DataGenerator
from model import PerspectiveTransformer


# Read and process the data into a dataframe
data_path, dst_path = 'data', 'data/images'
df = read_hdf5_file(data_path, dst_path)

# Split dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print(f"size train: {len(train_df)}, size test: {len(test_df)}") 

# Create and compile the model
resnet_model = ResNet50(include_top=True, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
pt = PerspectiveTransformer(resnet_model)
model = pt.build()

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Create data generators
batch_size = 32
train_generator = DataGenerator(train_df, batch_size=batch_size)
test_generator = DataGenerator(test_df, batch_size=batch_size)

# Train the model
epochs = 100
model.fit(train_generator, epochs=epochs)

# Evaluate the model on the test set
evaluate_metrics(model, test_df)
