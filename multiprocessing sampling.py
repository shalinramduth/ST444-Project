import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
from random import *

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


#timing how long it takes
start_time = time.time()

listofsamples = []

#function for drawing samples
def f(n):
    return df.sample(n=n,replace=True)


if __name__ == "__main__":
    p=Pool(processes = 3)
    inputs = [300]*10000 #brackets is input, outer is number of samples
    pr = p.map(f,inputs)
    listofsamples = pr
    print(pr[10])
	#I only printed one value from the list because of the 
	#length and number of the samples


print("--- %s seconds ---" % (time.time() - start_time))