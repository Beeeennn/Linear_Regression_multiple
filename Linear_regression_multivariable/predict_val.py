import numpy as np

def load_params():
    f = open("parameters.txt","r")
    w,b = (f.read()).split("|")
    w = str(w)[1:-1]
    w = w.split(", ")
    f.close
    print(w)
    wint = []
    for item in w:
        wint += [float(item)]
    b = float(b)
    return wint,b

def predict_val(x):
    w,b = load_params()
    prediction = np.dot(w,x)+b
    return prediction
print(predict_val([4205,2498016,33.5,10.9,5905,6179,173821,81.9,25.3,8.1,6.1,19588,48931]))