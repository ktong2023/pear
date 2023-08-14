import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers
import random
from library import conv_bn, dense_bn, tnet, NUM_POINTS, NUM_CLASSES, BATCH_SIZE
tf.random.set_seed(1234)
DATA_DIR = "./3-apple_point_clouds"



def parse_dataset(num_points):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in train_files:
            uneven_pc = np.loadtxt(f, delimiter=" ").tolist()
            if len(uneven_pc) < num_points:
                continue
            even_pc = random.sample(uneven_pc, num_points)
            train_points.append(np.array(even_pc)[:,:6])
            train_labels.append(i)
        for f in test_files:
            uneven_pc = np.loadtxt(f, delimiter=" ").tolist()
            if len(uneven_pc) < num_points:
                continue
            even_pc = random.sample(uneven_pc, num_points)
            test_points.append(np.array(even_pc)[:,:6])
            test_labels.append(i)
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

inputs = keras.Input(shape=(NUM_POINTS, 6))

x = tnet(inputs, 6)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)


model.fit(train_dataset, epochs=20, validation_data=test_dataset)

model.save_weights("./saved_model/apple_ripeness_predictor_model")
print("Model fitted and saved successfully!")