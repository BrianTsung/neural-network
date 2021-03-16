from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import filedialog
import random

url = ""
filename = ""


def openfile():
    global url, filename
    url = filedialog.askopenfilename(title="Select file")
    filename = url.split("/")[-1]
    filename = filename.split(".")[0]
    label_filename = tk.Label(window, width=12, text=filename + ".txt", fg='black')
    label_filename.place(x=170, y=10)


def readfile():
    read = open(url)
    file = read.readlines()
    read.close()
    return file


yy = 0


def preprocessing():
    file = readfile()
    data_x = []
    data_y = []
    lr = float(entry_lr.get())
    epoch = int(entry_epoch.get())
    for i in range(len(file)):
        ary = file[i].split()
        data_x.append((float(ary[0]), float(ary[1])))
        data_y.append(float(ary[2]))

    # legend classification
    legend = []
    for item in data_y:
        if item not in legend:
            legend.append(item)
    legend.sort()
    for i in range(len(data_y)):
        if data_y[i] == legend[0]:
            data_y[i] = -1
        else:
            data_y[i] = 1
    legend[0] = -1
    legend[1] = 1
    weight = [-1, 0, 1]

    # training & testing
    sample_rate = round(len(data_y)*(2/3))
    train_x, train_y = zip(*random.sample(list(zip(data_x, data_y)), sample_rate))
    test_x = []
    test_y = []
    train_x = list(train_x)
    for i in range(len(data_x)):
        if data_x[i] not in train_x:
            test_x.append(data_x[i])
            test_y.append(data_y[i])
    train_x1, train_x2 = zip(*train_x)
    test_x1, test_x2 = zip(*test_x)

    data_train1 = []
    data_train2 = []
    for i in range(len(train_y)):
        if train_y[i] == legend[0]:
            data_train1.append(train_x[i])
        else:
            data_train2.append(train_x[i])
    train1_x1, train1_x2 = zip(*data_train1)
    train2_x1, train2_x2 = zip(*data_train2)

    data_test1 = []
    data_test2 = []
    global yy
    for i in range(len(test_y)):
        if test_y[i] == legend[0]:
            yy = legend[0]
            data_test1.append(test_x[i])
        else:
            yy = legend[1]
            data_test2.append(test_x[i])
    if len(data_test1) != 0 and len(data_test2) == 0:
        test1_x1, test1_x2 = zip(*data_test1)
        return lr, epoch, weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2
    elif len(data_test2) != 0 and len(data_test1) == 0:
        test2_x1, test2_x2 = zip(*data_test2)
        return lr, epoch, weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test2_x1, test2_x2
    else:
        test1_x1, test1_x2 = zip(*data_test1)
        test2_x1, test2_x2 = zip(*data_test2)
        return lr, epoch, weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2, test2_x1, test2_x2


# algorithm
def perceptron():
    if filename == "test" or filename == "perceptron1" or filename == "perceptron2":
        lr, epoch, weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2 = preprocessing()
        case = 1
    else:
        case = 2
        lr, epoch, weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2, test2_x1, test2_x2 = preprocessing()
    epoch_count = 0
    for i in range(len(train_y)*epoch):
        if i > 0 and i % len(train_y) == 0:
            epoch_count += len(train_y)
        i = i - epoch_count
        output = weight[0]*-1+weight[1]*train_x1[i]+weight[2]*train_x2[i]
        if output > 0:
            output = 1
        else:
            output = -1

        if output > train_y[i]:
            weight[0] = weight[0] - lr * -1
            weight[1] = weight[1] - lr * train_x1[i]
            weight[2] = weight[2] - lr * train_x2[i]

        elif output < train_y[i]:
            weight[0] = weight[0] + lr * -1
            weight[1] = weight[1] + lr * train_x1[i]
            weight[2] = weight[2] + lr * train_x2[i]
    weight[0] = round(weight[0], 3)
    weight[1] = round(weight[1], 3)
    weight[2] = round(weight[2], 3)
    display_weight0 = tk.Label(window, width=10, text=weight[0], fg='black')
    display_weight0.place(x=120, y=140)
    display_weight1 = tk.Label(window, width=10, text=weight[1], fg='black')
    display_weight1.place(x=120, y=170)
    display_weight2 = tk.Label(window, width=10, text=weight[2], fg='black')
    display_weight2.place(x=120, y=200)

    y = np.linspace(min(train_x2), max(train_x2), 2)
    if weight[1] != 0:
        x = (-1*weight[2]*y + weight[0]) / weight[1]
    else:
        x = weight[0] / weight[2]
        x = np.array([x, x])

    # draw
    f_plot.clear()
    f_plot.plot(x, y)
    legend_train1 = f_plot.scatter(train1_x1, train1_x2, c='r', marker='o', s=8)
    legend_train2 = f_plot.scatter(train2_x1, train2_x2, c='r', marker='x', s=8)
    global yy
    if case == 1:
        if yy == -1:
            legend_test1 = f_plot.scatter(test1_x1, test1_x2, c='b', marker='o', s=8)
            f.legend([legend_train1, legend_train2, legend_test1],
                     ["train:-1", "train:1", "test:-1"], loc='upper left', prop={'size': 5})
        elif yy == 1:
            legend_test2 = f_plot.scatter(test1_x1, test1_x2, c='b', marker='x', s=8)
            f.legend([legend_train1, legend_train2, legend_test2],
                     ["train:-1", "train:1", "test:1"], loc='upper left', prop={'size': 5})
    elif case == 2:
        legend_test1 = f_plot.scatter(test1_x1, test1_x2, c='b', marker='o', s=8)
        legend_test2 = f_plot.scatter(test2_x1, test2_x2, c='b', marker='x', s=8)
        f.legend([legend_train1, legend_train2, legend_test1, legend_test2],
                 ["train:-1", "train:1", "test:-1", "test:1"], loc='upper left', prop={'size': 5})
    f_plot.set_xlabel('X1', fontsize=8)
    f_plot.set_ylabel('x2', fontsize=8)
    canvas.draw()
    accuracy(weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y)


def accuracy(weight, train_x1, train_x2, train_y, test_x1, test_x2, test_y):
    print("weight:", weight)
    train_true = 0
    for i in range(len(train_y)):
        print(train_x1[i], train_x2[i])
        output = weight[0] * -1 + weight[1] * train_x1[i] + weight[2] * train_x2[i]
        print("output", output)
        if output > 0:
            output = 1
        else:
            output = -1

        if output == train_y[i]:
            train_true += 1
        train_acc = round(train_true*100 / len(train_y), 1)
    display_train_acc = tk.Label(window,  width=10, text=str(train_acc)+"%", fg='black')
    display_train_acc.place(x=120, y=250)

    test_true = 0
    for i in range(len(test_y)):
        print(test_x1[i], test_x2[i])
        output = weight[0] * -1 + weight[1] * test_x1[i] + weight[2] * test_x2[i]
        print("output", output)
        if output > 0:
            output = 1
        else:
            output = -1
        if output == test_y[i]:
            test_true += 1
        test_acc = round(test_true * 100 / len(test_y), 1)
    display_test_acc = tk.Label(window, width=10, text=str(test_acc) + "%", fg='black')
    display_test_acc.place(x=120, y=280)


# gui
window = tk.Tk()
window.title("NN HW1-109522050")
window.geometry('750x450')

# select dataset
label_data = tk.Label(window, text="select dataset", fg='black')
label_data.place(x=30, y=10)
button_data = tk.Button(window, text='select', fg='blue', command=openfile)
button_data.place(x=120, y=10)

# input lr
label_lr = tk.Label(window, text="learning rate", fg='black')
label_lr.place(x=30, y=60)
input_lr = tk.DoubleVar()
input_lr.set(0.8)
entry_lr = tk.Entry(window, width=20, textvariable=input_lr)
entry_lr.place(x=120, y=60)

# input epoch
label_epoch = tk.Label(window, text="epoch", fg='black')
label_epoch.place(x=30, y=90)
input_epoch = tk.IntVar()
input_epoch.set(1)
entry_epoch = tk.Entry(window, width=20, textvariable=input_epoch)
entry_epoch.place(x=120, y=90)

# display weight
label_weight0 = tk.Label(window, text="weight0", fg='blue')
label_weight0.place(x=30, y=140)
label_weight1 = tk.Label(window, text="weight1", fg='blue')
label_weight1.place(x=30, y=170)
label_weight2 = tk.Label(window, text="weight2", fg='blue')
label_weight2.place(x=30, y=200)

# display accuracy
label_train_acc = tk.Label(window, text="training_acc", fg='blue')
label_train_acc.place(x=30, y=250)
label_test_acc = tk.Label(window, text="testing_acc", fg='blue')
label_test_acc.place(x=30, y=280)

f = Figure(figsize=(3, 3), dpi=150)
f.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

f_plot = f.add_subplot(111)
f_plot.tick_params(labelsize=8)
canvas = FigureCanvasTkAgg(f, window)
canvas.get_tk_widget().place(x=300, y=0)

# submit botton
button = tk.Button(window, height=2, width=8, text='run', fg='blue', command=perceptron)
button.place(x=120, y=350)

window.mainloop()
