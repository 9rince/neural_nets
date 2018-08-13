import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

new_model = tf.keras.models.load_model('dig_recog.model')
predictions = new_model.predict(x_test)
pred_2 = [np.argmax(i) for i in predictions]
p_u,p_c = np.unique(pred_2, return_counts=True)
y_u,y_c = np.unique(y_test, return_counts=True)
print("predicted ->",dict(zip(p_u,p_c)))
print("original  ->",dict(zip(y_u,y_c)))

# plt.imshow(x_test[0],cmap=plt.cm.binary)
# plt.show()
