# import module
import tensorflow as tf

# Load and prepare the MINIST dataset
# Convert the sample from integers to floating-point numbers
mnist = tf.keras.dataset.minist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(type(x_train), type(y_train))
print(type(x_train.shape, x_test.shape)

# Build tf.keras. sequintal model by stacking layers
# Choose an optimizer and loss function for training

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train and evaluate the performance of the model
model.fit(x_train, y_train, epoch=5)
model.evaluate(x_test, y_test, verbose=2)
