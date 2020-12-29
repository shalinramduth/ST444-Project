import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time

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


times = [] #varying the inputs so that they are equally spaced on a log scale for plotting results

for x in range(10):
    times.append(round(10**(4+0.2*x)))

def f(n):
    return df.sample(n,replace=True).mean()
s = []
t = 0

for x in times: #n in parallel computing function
    start_time = time.time()
    q = []
    for y in range(1000): #m in parallel computing function   
        t=f(x)
        q.append(t)
    s.append((time.time() - start_time,y))

print(s)

#Non parallel times and sample sizes: [x]*1000
#[(4.21012282371521, 10000), (5.13896632194519, 15849), (7.953479051589966, 25119), (10.965460777282715, 39811), (14.028553247451782, 63096), (20.602095127105713, 100000), (54.665878772735596, 158489), (58.047602891922, 251189), (90.48633790016174, 398107), (138.72087740898132, 630957)]

#Non parallel times and sample lengths: [1000]*x
#[(0.022881031036376953, 6310), (0.02356123924255371, 6310), (0.025014400482177734, 6310), (0.017755746841430664, 6310), (0.03243374824523926, 6310), (0.024686098098754883, 6310), (0.02382040023803711, 6310), (0.019107341766357422, 6310), (0.031991004943847656, 6310), (0.025196552276611328, 6310)]
