import numpy as np
import pandas as pd

data = pd.read_csv('data/result.csv')
nn_re = np.array(pd.read_csv('data/result.csv')['happiness'])
tree_re = np.array(pd.read_csv('data/treeresult.csv')['happiness'])

a = 0.98
b = 1-a
print(data.columns)
prediction = a*tree_re + b*nn_re

last_re = pd.DataFrame()
prediction[prediction<0] = 0
last_re['id'] = data.id
last_re['happiness'] = prediction
last_re.to_csv('predictions.csv',index= False)