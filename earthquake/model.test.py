import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

arr = np.array([[23.46,94.61,19]])
output = model.predict(arr)

print(output[0])