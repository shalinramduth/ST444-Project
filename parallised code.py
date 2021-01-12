# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:21:59 2021

@author: Chenx
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn import tree
import concurrent.futures as cf
import os

os.chdir("C:\\Users\\Chenx\\Desktop\\LSE master\\ST444\\project")
file = "ObesityDataSet_raw_and_data_sinthetic.csv"
df = pd.read_csv(file)

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
        

x = df[df.columns[0:16]]; y = df[df.columns[16]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1)

#d = df.sample(n=300,replace=True)

times = list(np.linspace(100, 1414, num=10)) # create equal spaced sample size


s = [] #list of times taken
s1 = []
y_test = np.array(list(y_test))
#function being used in each core
def f(n):
    X = x_train.sample(n=n,replace=True) # sample from training dataset
    Y = y_train.loc[list(X.index)] #get corresponding Y
    
    # fit the decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y) # fit the decision tree with the sample drawn
    
    # predict the obesity level with the decision tree
    y_predict = clf.predict(x_test)
    
    return (y_predict)
    
 

def mult(n,m):
    #n is how large each sample is, m is how many samples are taken
    inputs = [n]*m

    if __name__ == "__main__":
        #uses current.futures module instead of multiprocessing
        with cf.ProcessPoolExecutor() as ex:
            #timing
            start = time.perf_counter()

            #uses map to map the function to the inputs and put into results list
            results = ex.map(f, inputs)
            
            #predicting the class
            pred = np.rint((np.mean([x for x in results], axis = 0)))

            #printing accuracy
            accu = accuracy_score(list(y_test),pred)

            finish = time.perf_counter()

            #time taken
            t = finish - start
            
            return (accu, t)

# create a dataframe the store the result
df_time = pd.DataFrame(columns = ['sample size', 'time taken (parallel)', 'accuracy (parallel)'])
df_time['sample size'] = times

for row in df_time.iterrows():
    row['accuracy (parallel)'] = mult(row['sample size'], 500)[0]
    row['time taken (parallel)'] = mult(row['sample size'], 500)[1]

