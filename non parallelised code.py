import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
from sklearn import tree
from sklearn.metrics import accuracy_score


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

times = [] #varying the inputs so that they are equally spaced on a log scale for plotting results

for x in range(5):
    times.append(round(10**(4+0.2*x)))

def f(n):
    X = x_train.sample(n=n,replace=True) # sample from training dataset
    Y = y_train.loc[list(X.index)] #get corresponding Y
    
    # fit the decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y) # fit the decision tree with the sample drawn
    
    # predict the obesity level with the decision tree
    y_predict = clf.predict(x_test)
    
    return y_predict
s = []
t = 0

for x in times: #n in parallel computing function
    start = time.perf_counter()
    q = []
    for y in range(500): #m in parallel computing function   
        t=f(x)
        q.append(t)
    #s.append((time.time() - start_time))
    pred = np.rint((np.mean(q, axis = 0)))
    print('accuracy score: ',accuracy_score(list(y_test),pred))
    finish = time.perf_counter()

    #returns the time taken
    print('time taken: ', finish - start)



