import numpy as np 
import pandas as pd 
import keras 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential
from keras.layers import Dense , Dropout 
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier

# data loading
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[: , 3:13].values
y = dataset.iloc[: , 13].values

#Encoding the labels 
labelencoder_X_1 = LabelEncoder()
X[: , 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[: , 2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#Splitting into training AND test test
train_X , test_X , train_y , test_y = train_test_split(X , y , test_size = 0.2 , random_state = 0)

#data preprocessing 
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)

def build_model():
	classifier = Sequential()
	classifier.add(Dense(output_dim = 8 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 11))
	classifier.add(Dropout(p = 0.1))
	classifier.add(Dense(output_dim = 6 , kernel_initializer = 'uniform'  , activation = 'relu'))
	classifier.add(Dropout(p = 0.1))
	classifier.add(Dense(output_dim = 6 , kernel_initializer = 'uniform' , activation = 'relu'))
	classifier.add(Dense(output_dim = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
	return classifier


classifier = KerasClassifier(build_fn = build_model , batch_size = 15 , epochs = 200)
accuracies = cross_val_score(estimator = classifier , X = train_X , y = train_y , cv = 10 , n_jobs = -1)

accuracy = accuracies.mean()
variance = accuracies.std()

print("Train set accurcy is"+str(accuracy))
print("Variance is"+str(variance))







