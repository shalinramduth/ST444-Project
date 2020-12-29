import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
import multiprocessing

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

#d = df.sample(n=300,replace=True)

times = [] #varying the inputs so that they are equally spaced on a log scale for plotting results
for x in range(10):
    times.append(round(10**(4+0.2*x)))

s = [] #list of times taken

#function being used in each core
def f(n): 
    return df.sample(n,replace=True).mean()

cores = multiprocessing.cpu_count()
print(cores)

def mult(n,m):

    start_time = time.time()

    if __name__ == "__main__":
        p=Pool(processes = cores-1)
        inputs = [n]*m
        pr = p.map(f,inputs)
    s.append((time.time() - start_time,x)) #final time of each sampling
    

for x in times:
    mult(x,1000)
print(s)

#Parallel times and sample sizes with 3 cores: [n]*10000
#[(6.559086561203003, 10000), (6.838163137435913, 15849), (8.188640117645264, 25119), (10.099955081939697, 39811), (12.616981267929077, 63096), (17.90579652786255, 100000), (23.442461252212524, 158489), (31.79633903503418, 251189), (46.863340616226196, 398107), (70.94382572174072, 630957)]

#Parallel times and sample sizes with 2 cores: [n]*10000
#[(6.3229265213012695, 10000), (6.325735092163086, 15849), (7.457021236419678, 25119), (8.776432752609253, 39811), (11.506320476531982, 63096), (15.59110426902771, 100000), (24.394813776016235, 158489), (36.230053424835205, 251189), (63.184372901916504, 398107), (83.46709561347961, 630957)]


#Parallel times and sample lengths with 3 cores: [1000]*x
#[(3.7297873497009277, 100), (3.294684886932373, 158), (3.8836770057678223, 251), (3.329380750656128, 398), (4.534084320068359, 631), (4.646569013595581, 1000), (5.722285270690918, 1585), (7.475570917129517, 2512), (10.884510517120361, 3981), (17.004668474197388, 6310)]

#Parallel times and sample lengths with 2 cores: [1000]*x
#[(4.242950439453125, 100), (3.5940287113189697, 158), (3.8869664669036865, 251), (4.575963020324707, 398), (5.410073280334473, 631), (6.677933931350708, 1000), (9.432047605514526, 1585), (13.052958250045776, 2512), (14.90382719039917, 3981), (19.030319452285767, 6310)]
