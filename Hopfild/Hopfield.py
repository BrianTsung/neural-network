import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
import random

url = ""
filename = ""
url_test = ""
filename_test = ""

def openfile():
    global url, filename
    url = filedialog.askopenfilename(title="Select training file")
    filename = url.split("/")[-1]
    filename = filename.split(".")[0]
    label_filename = tk.Label(
        window, width=16, text=filename + ".txt", fg='black')
    label_filename.place(x=230, y=10)

def openfile_test():
    global url_test, filename_test
    url_test = filedialog.askopenfilename(title="Select test file")
    filename_test = url_test.split("/")[-1]
    filename_test = filename_test.split(".")[0]
    label_filename_test = tk.Label(
        window, width=16, text=filename_test + ".txt", fg='black')
    label_filename_test.place(x=230, y=40)

def read_data(url):
    file = open(url)
    read = file.readlines()
    file.close()
    return read

data_amount = 1
row_amount = 0
def preprocessing(read_data):
    data = []
    global data_amount, row_amount
    data_amount = 1
    for row in read_data:
        row = list(row)
        for item in range(len(row)):
            if row[item] == " ":
                row[item] = -1
            elif row[item]  == "\n":
                row.remove(row[item])
            else:
                row[item] = int(row[item])
        if len(row) != 0:
            data.append(row)
        else:
            data_amount += 1 
    finaldata = []
    row_amount = int(len(data)/data_amount)
    for i in range(data_amount):
        mergelist = []
        for j in range(row_amount):
            mergelist += data[j+i*row_amount] 
        finaldata.append(mergelist)
    finaldata = np.array(finaldata)
    return finaldata

def hopfield():
    read = read_data(url)
    finaldata = preprocessing(read)
    matrix_size = len(finaldata[0])
    column_amount = int(matrix_size/row_amount)
    matrix_sum = np.zeros((matrix_size, matrix_size))
    testno = int(entry_test_no.get()) - 1
   
    fig_train = Figure(figsize=(5, 5))
    fig_train.clear()
    reshape_train_data = finaldata[testno].reshape(row_amount, column_amount)
    axes = fig_train.add_subplot(2,1,1)
    axes.set_title("train_figure_original", pad=20)
    caxes = axes.matshow(reshape_train_data)
    fig_train.colorbar(caxes) 
    if filename == "Basic_Training":
        for i in range(data_amount):
            randomlist = []
            for j in range(0,10):
                n = random.randint(0,matrix_size-1)
                randomlist.append(n)
            for randno in randomlist:
                if finaldata[i][randno] == -1:
                    finaldata[i][randno] = 1
                else:
                    finaldata[i][randno] = -1

        reshape_train_data = finaldata[testno].reshape(row_amount, column_amount)
        axes = fig_train.add_subplot(2,1,2)
        axes.set_title("train_figure_noise", pad=20)
        caxes = axes.matshow(reshape_train_data)
        fig_train.colorbar(caxes) 
        
    fig_train.tight_layout() 
    canvas = FigureCanvasTkAgg(fig_train, window)
    canvas.get_tk_widget().place(x=550, y=100)
    canvas.draw()

    for i in range(data_amount):
        expand_dim_matrix = np.expand_dims(finaldata[i], axis=0)
        matrix_dot = (expand_dim_matrix.T).dot(expand_dim_matrix)
        matrix_sum = np.add(matrix_sum, matrix_dot) 

    I = np.identity(matrix_size)
    I = I * (data_amount/matrix_size)
    matrix_sum = matrix_sum * (1/matrix_size)
    W = np.subtract(matrix_sum, I)
    theta = np.sum(W, axis=0)
    read = read_data(url_test)
    test_data = preprocessing(read)
    test_data = np.expand_dims(test_data[testno], axis=0)

    fig = Figure(figsize=(7, 7)) 
    for count in range(1):
        subplot_no = 1
        fig.clear()
        bonus_list = []
        for i in range(matrix_size):
            output = (W[i].dot(test_data.T)) - theta[i]
            if filename == "Basic_Training":
                if output > 0:
                    test_data[0][i] = 1
                elif output < 0:
                    test_data[0][i] = -1
            else:
                if output > 0:
                    bonus_list.append(1)
                elif output < 0:
                    bonus_list.append(-1)
            if filename == "Basic_Training":
                if i%14 == 0 or i==107:
                    reshape_test_data = test_data.reshape(row_amount, column_amount)
                    axes = fig.add_subplot(3,3,subplot_no)
                    axes.set_title("iteration" + str(i+1), pad=20)
                    caxes = axes.matshow(reshape_test_data)
                    fig.colorbar(caxes) 
                    subplot_no += 1
        if filename == "Bonus_Training":
            bonus_list = np.array(bonus_list)
            bonus_list = np.expand_dims(bonus_list, axis=0)
            test_data = bonus_list
            reshape_test_data = bonus_list.reshape(row_amount, column_amount)
            axes = fig.add_subplot(1,1,1)
            caxes = axes.matshow(reshape_test_data)
            fig.colorbar(caxes)
    fig.tight_layout() 
    canvas = FigureCanvasTkAgg(fig, window)
    canvas.get_tk_widget().place(x=0, y=100)
    canvas.draw()

# gui
window = tk.Tk()
window.title("NN HW3-109522050")
window.geometry('1050x800')

# select dataset
label_data = tk.Label(window, text="select training dataset", fg='black')
label_data.place(x=30, y=10)
button_data = tk.Button(window, text='select', fg='blue', command=openfile)
button_data.place(x=180, y=10)
label_data_test = tk.Label(window, text="select testing dataset", fg='black')
label_data_test.place(x=30, y=40)
button_data_test = tk.Button(window, text='select', fg='blue', command=openfile_test)
button_data_test.place(x=180, y=40)

# input test data number
label_test_no = tk.Label(window, text="test data number(Basic:1~3  Bonus:1~15)", fg='black')
label_test_no.place(x=30, y=70)
input_test_no = tk.IntVar()
input_test_no.set(1)
entry_test_no = tk.Entry(window, width=20, textvariable=input_test_no)
entry_test_no.place(x=280, y=70)

#submit button
button = tk.Button(window, height=2, width=8, text='run',
                   fg='blue', command=hopfield)
button.place(x=600, y=10)
window.mainloop()