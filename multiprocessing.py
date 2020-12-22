
# import the packages
import os
import pandas as pd
from sklearn import tree
import random
import numpy as np
from sklearn.model_selection import train_test_split
import time
from multiprocessing import Pool

os.chdir("C:\\Users\\Chenx\\Desktop\\LSE master\\ST444\\project")

#import the dataset and have a look of the data
file = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(file)
print(df.info())

# pre-process and clean the data
'''77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was 
collected directly from users through a web platform. According to Data Brief'''
# Round age, and Weight to integer; Round Height to 2 decimal places; As for other float64-type varialbes, they should be integer
integer = ["Age", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
df[integer] = df[integer].apply(np.int64)
df = df.round({"Height": 2})


# convert categorical variables
for column in df.columns:
    if df[column].dtypes == "object":
        df[column] = pd.factorize(df[column])[0]
        

# define variates x, and response variable y
x = df[df.columns[0:16]]; y = df[df.columns[16]]

# split the data into training and testing dataset for cross validation
# training: 2/3; testing: 1/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)

# use decision tree to predict obesity level
start_time = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# predict the obesity level with the decision tree
y_predict = clf.predict(x_test)

# prediction accuracy
lis = []
for (y_pre, y_tes) in zip(y_predict, y_test):
    if y_pre != y_tes:
        lis.append(1)
    else:
        lis.append(0)

print("The Error rate is", sum(lis)/len(y_test))
print("--- %s seconds ---" % (time.time() - start_time))



#timing how long it takes
start_time = time.time()

prediction = []

# define the function that returns prediction 
def f(n):
    X = x_train.sample(n=n,replace=True) # sample from training dataset
    Y = y_train.loc[list(X.index)] #get corresponding Y
    
    # fit the decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y) # fit the decision tree with the sample drawn
    
    # predict the obesity level with the decision tree
    y_predict = clf.predict(x_test)
    
    return (y_predict)


if __name__ == "__main__":
    p=Pool(processes = 3)
    inputs = [1414]*10000 #brackets is input, outer is number of samples
    pr = p.map(f,inputs)
    prediction = pr
	#I only printed one value from the list because of the 
	#length and number of the samples


print("--- %s seconds ---" % (time.time() - start_time))


# to be completed 
# convert prediction from list to numpy array
prediction = np.array(prediction)

# prediction with bagging
bagging_pred = np.rint((np.mean(prediction, axis = 0)))


