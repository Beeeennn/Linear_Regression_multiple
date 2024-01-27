from numpy import genfromtxt
import pandas as pd


def convert_csv_single(filename,y_column,ignore):

    my_data = genfromtxt(filename, delimiter=',')

    x = []
    y = []

    for training_example in my_data:

        xn =  []

        for i in range(len(training_example)):

            if i in ignore:
                continue
            elif i == y_column:
                y += [training_example[i]]
            else:
                x += [training_example[i]]

    return x[1:],y[1:]

def convert_csv_multiple(filename,y_column,ignore):

    my_data = genfromtxt(filename, delimiter=',')

    x = []
    y = []

    for training_example in my_data:

        xn =  []

        for i in range(len(training_example)):

            if i in ignore:
                continue
            elif i == y_column:
                y += [training_example[i]]
            else:
                xn += [training_example[i]]

        x += [xn]

    return x[1:],y[1:]


#x,y = convert_csv_single("Salary_dataset.csv",2,0)
"""print(x)
print("")
print(y)"""