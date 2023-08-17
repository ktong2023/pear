import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers
import random
from library import conv_bn, dense_bn, tnet, NUM_POINTS, NUM_CLASSES, BATCH_SIZE, CLASS_MAP



print("Please input the folder path to the files you wish to predict:")
dir_path = input()
"""
./advripe_test - Use this file path to test 10 point clouds of ripe apples
./unripe_test - Use this file path to test 10 point clouds of unripe apples
"""

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

model.load_weights('./saved_model/apple_ripeness_predictor_model').expect_partial()

def test_input (dir_path):
    INPUT_DIR = dir_path
    input_files = glob.glob(os.path.join(INPUT_DIR, "*"))

    input_points = []

    for f in input_files:
        uneven_pc = np.loadtxt(f, delimiter=" ").tolist()
        if len(uneven_pc) < NUM_POINTS:
            continue
        even_pc = random.sample(uneven_pc, NUM_POINTS)
        input_points.append(np.array(even_pc)[:,:6])

    input_dataset = tf.data.Dataset.from_tensor_slices((input_points))
    input_dataset = input_dataset.shuffle(len(input_points)).batch(BATCH_SIZE)

    preds = model.predict(input_dataset)

    # Take the prediction with the highest probability and disregard values less than -1
    preds = tf.math.argmax(preds, -1)
    print(preds)
    return preds

output_predictions = test_input(dir_path).numpy()
answer = dir_path.replace(dir_path[:2], '')
for i in range(10):
    print("prediction: ", CLASS_MAP[output_predictions[i]], " Answer: ", answer)