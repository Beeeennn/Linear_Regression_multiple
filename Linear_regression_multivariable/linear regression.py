import numpy as np
import matplotlib.pyplot as plt
from Utils import *

def compute_cost(x,y,w,b,lambda_):

    m,n = np.shape(x)
    cost = 0
    for i in range(m):
        cost += (np.dot(w,x[i])+b-y[i])**2
    cost = (cost/(m*2))
    
    reg_term = 0
    for j in range(len(w)):
        reg_term += (w[j])**2
    reg_term = (reg_term*lambda_)/(2*m)
    cost += reg_term

    print(cost)

    return cost

def predict(x,w,b):
    p=np.dot(x,w) + b
    return p



def grad_desc(x,y,w,b,lambda_):
    m,n = np.shape(x)
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        for j in range(n):
            dj_dw[j] += (((np.dot(w,x[i])+b-y[i])*x[i][j])+(lambda_*w[j]))/m
        dj_db += (np.dot(w,x[i])+b-y[i])
    dj_db /= m

    return dj_dw,dj_db


def update_values(x,y,w,b,rate,lambda_):
    dj_dw,dj_db = grad_desc(x,y,w,b,lambda_)
    wt = w
    for j in range(len(w)):
        wt[j] = w[j] - rate*(dj_dw[j])
    bt = b - rate*(dj_db)
    w = wt
    b = bt
    return w,b

def train_model(x,y,w,b,rate,itterations,lambda_):
    costs = []
    itts = []
    for i in range(itterations):
        if i%10 == 0:
            cost = compute_cost(x,y,w,b,lambda_)
            print(f"Cost after itteration {i}: {cost}")
            costs += [cost]
            itts += [i]
        w,b = update_values(x,y,w,b,rate,lambda_)
    plt.plot(itts,costs)
    plt.show()

    savedvals = str(list(w))+"|"+str(b)
    f = open("parameters.txt","w")
    f.write(savedvals)
    f.close()
    print("Trained!")
    return(w,b)

def predict_value(inp,w,b):
    return w*inp+b

x,y = convert_csv_multiple("dataset.csv",16,[0,1,2])
m,n = np.shape(x)
w = np.zeros((n,))
b = 0
print(f"X is {x[:10]} \n\n Y is {y[:10]}")
w,b = train_model(x,y,w,b,0.00000000000003,400,0)

