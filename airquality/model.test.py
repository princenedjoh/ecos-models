import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
rfr = pickle.load(open('rfr.pkl', 'rb'))

arr = np.array([[213.62,0.25,2.23,92.98,1.33,0.5,0.6,0.49]])
output = model.predict(arr)
output2 = rfr.predict(arr)

print(output[0])
print(output2[0])