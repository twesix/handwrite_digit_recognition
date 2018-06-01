import keras
import numpy as np
from keras.models import model_from_json

x_test = []
for i in range(24):
    x_test.append(np.random.random(24))
x_test = np.array(x_test)
print(x_test.shape)

with open('model_digit.json') as f:
    json_str = f.read()

model = model_from_json(json_str)
model.load_weights('model_digit.h5')
print(model.predict(x_test))


def recognize(array):
    pass


