import numpy as np
from keras.models import load_model, model_from_json

model = load_model('trained_model.h5')
# with open('model_digit.json') as f:
#     model1 = model_from_json(f.read())
# model.load_weights('model_digit.h5')


def recognize(array):

    x_input = []
    for i in range(28):
        dim2 = []
        for j in range(28):
            dim2.append([0])
        x_input.append(dim2)

    for item in array:
        x_input[item[0]][item[1]] = [item[2]]

    x_input = np.array([x_input]).astype('float32')
    output = model.predict(x_input)[0].tolist()
    return output.index(max(output))


if __name__ == '__main__':
    x_test = [
        [12, 0, 0.8],
        [12, 1, 0.8],
        [12, 2, 0.8],
        [12, 3, 0.8],
        [12, 4, 0.8],
        [12, 5, 0.8],
        [12, 6, 0.8],
        [12, 7, 0.8],
        [12, 8, 0.8],
        [12, 9, 0.8],
        [12, 10, 0.8],
        [12, 11, 0.8],
        [12, 12, 0.8],
        [12, 13, 0.8],
        [12, 14, 0.8],
        [12, 15, 0.8],
        [12, 16, 0.8],
        [12, 17, 0.8],
        [12, 18, 0.8],
        [12, 19, 0.8],
        [12, 20, 0.8],
        [12, 21, 0.8],
        [12, 22, 0.8],
        [12, 23, 0.8],
        [13, 0, 0.8],
        [13, 1, 0.8],
        [13, 2, 0.8],
        [13, 3, 0.8],
        [13, 4, 0.8],
        [13, 5, 0.8],
        [13, 6, 0.8],
        [13, 7, 0.8],
        [13, 8, 0.8],
        [13, 9, 0.8],
        [13, 10, 0.8],
        [13, 11, 0.8],
        [13, 12, 0.8],
        [13, 13, 0.8],
        [13, 14, 0.8],
        [13, 15, 0.8],
        [13, 16, 0.8],
        [13, 17, 0.8],
        [13, 18, 0.8],
        [13, 19, 0.8],
        [13, 20, 0.8],
        [13, 21, 0.8],
        [13, 22, 0.8],
        [13, 23, 0.8],
    ]
    print(recognize(x_test))
