# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('insurance.csv')


X = dataset.iloc[:, :6]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'male':1,'female':2, 'northeast':1,'southeast':2,'southwest':3,'northwest':4, 'yes':1,'no':0}
    return word_dict[word]

X['sex'] = X['sex'].apply(lambda x : convert_to_int(x))
X['region'] = X['region'].apply(lambda x : convert_to_int(x))
X['smoker'] = X['smoker'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#We will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data

regressor.fit(X.values, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
