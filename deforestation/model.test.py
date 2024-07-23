import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

arr = np.array([[306.75006103515625,1.0,22.0,86.0,69.0,29.0,0.0,25.0,88.0,77.0,36.0,1.0,3.0,4.640261173248291]])
output = model.predict(arr)

print(output[0])