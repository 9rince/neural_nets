import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt



mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print("\n  -> x_train contains {0} images and x_test contains {1} images".format(len(x_train),len(x_test)))
print("\n  -> each image is {0} x {1}".format(len(x_train[0][0]),len(x_train[0])))

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
# print("\n  -> label for first image is {}".format(y_train[0]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
val_loss, val_acc = model.evaluate(x_test, y_test)
print("\n ->  error {}".format(val_loss))
print("\n ->  accuracy {}".format(val_acc))

model.save('dig_recog.model')
