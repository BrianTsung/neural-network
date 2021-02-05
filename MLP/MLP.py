from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import random
import math

url = ""
filename = ""


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def openfile():
    global url, filename
    url = filedialog.askopenfilename(title="Select file")
    filename = url.split("/")[-1]
    filename = filename.split(".")[0]
    label_filename = tk.Label(
        window, width=12, text=filename + ".txt", fg='black')
    label_filename.place(x=170, y=10)


def readfile():
    read = open(url)
    file = read.readlines()
    read.close()
    return file


def preprocessing():
    file = readfile()
    data_x = []
    data_y = []
    if filename == "Number":
        for i in range(len(file)):
            ary = file[i].split()
            ary_y = ary.pop(25)
            ary_x = ary
            data_x.append(ary_x)
            data_y.append(ary_y)
    else:
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
            data_y[i] = 0
        else:
            data_y[i] = 1
    legend[0] = 0
    legend[1] = 1

    # training & testing
    sample_rate = round(len(data_y)*(2/3))
    train_x, train_y = zip(
        *random.sample(list(zip(data_x, data_y)), sample_rate))
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
    for i in range(len(test_y)):
        if test_y[i] == legend[0]:
            data_test1.append(test_x[i])
        else:
            data_test2.append(test_x[i])
    if len(data_test1) != 0 and len(data_test2) == 0:
        test1_x1, test1_x2 = zip(*data_test1)
        return train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2
    elif len(data_test2) != 0 and len(data_test1) == 0:
        test2_x1, test2_x2 = zip(*data_test2)
        return train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test2_x1, test2_x2
    else:
        test1_x1, test1_x2 = zip(*data_test1)
        test2_x1, test2_x2 = zip(*data_test2)
        return train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2, test2_x1, test2_x2


# algorithm
def perceptron():
    lr = float(entry_lr.get())
    epoch = int(entry_epoch.get())
    nnumber = int(entry_nnumber.get())
    weightlist = []
    for i in range(nnumber):
        wlist = []
        w = random.uniform(-1.5, 1)
        wlist.append(w)
        for j in range(2):
            wlist.append(w)
        weightlist.append(wlist)
    wlist = []
    for j in range(nnumber+1):
        w = random.uniform(-1.5, 1)
        wlist.append(w)
    weightlist.append(wlist)

    if filename == "test" or filename == "perceptron1" or filename == "perceptron2" or filename == "xor":
        train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2 = preprocessing()
        case = 1
    else:
        case = 2
        train_x1, train_x2, train_y, test_x1, test_x2, test_y, train1_x1, train1_x2, train2_x1, train2_x2, test1_x1, test1_x2, test2_x1, test2_x2 = preprocessing()
    epoch_count = 0
    epoch_text = 1
    """train_x1 = list(train_x1) + list(test_x1)
    train_x2 = list(train_x2) + list(test_x2)
    train_y = list(train_y) + list(test_y)"""
    progress_bar['maximum'] = len(train_y)*epoch
    text = tk.StringVar()
    text.set("epoch " + str(epoch_text) + "/" + str(epoch))
    progress_text = tk.Label(window, width=20, textvariable=text, fg='green', anchor = 'w')
    progress_text.place(x=30, y=430)
    for i in range(len(train_y)*epoch):
        if i == len(train_y)*epoch-1:
            text.set("waiting for the figure...")
        outputlist = []
        progress_bar['value'] = i+1
        progress_bar.update()
        if i > 0 and i % len(train_y) == 0:
            epoch_text += 1
            text.set("epoch " + str(epoch_text) + "/" + str(epoch))
            epoch_count += len(train_y)
        i = i - epoch_count

        # forward propagation
        for w_no in range(len(weightlist)-1):
            outputlist.append(sigmoid(
                weightlist[w_no][0] * -1 + weightlist[w_no][1] * train_x1[i] + weightlist[w_no][2] * train_x2[i]))
        finaloutput = weightlist[-1][0] * -1
        for o_no in range(0, len(outputlist)):
            finaloutput += weightlist[-1][o_no + 1] * outputlist[o_no]
        finaloutput = sigmoid(finaloutput)
        outputlist.append(finaloutput)
        if outputlist[-1] > 0.5:
            z = 1
        else:
            z = 0

        # back propagation
        errorlist = []
        if z != train_y[i]:
            finalerror = (train_y[i] - outputlist[-1]) * \
                outputlist[-1] * (1 - outputlist[-1])
            for x in range(len(outputlist)-1):
                errorlist.append(
                    outputlist[x] * (1 - outputlist[x]) * finalerror * weightlist[-1][x+1])
            errorlist.append(finalerror)
            for x in range(len(weightlist)-1):
                weightlist[x] = np.array(
                    weightlist[x]) + lr * errorlist[x] * np.array([-1, train_x1[i], train_x2[i]])
            outputlist.insert(0, -1)
            outputlist.remove(outputlist[-1])
            weightlist[-1] = np.array(weightlist[-1]) + \
                lr * errorlist[-1] * np.array(outputlist)
        for x in range(len(weightlist)):
            for y in range(len(weightlist[x])):
                weightlist[x][y] = round(weightlist[x][y], 3)

    # data transformation
    if nnumber == 2:
        t_train1_x1 = []
        t_train1_x2 = []
        t_train2_x1 = []
        t_train2_x2 = []
        t_test1_x1 = []
        t_test1_x2 = []
        t_test2_x1 = []
        t_test2_x2 = []
        for item_tr1 in range(len(train1_x1)):
            t_train1_x1.append(sigmoid(
                weightlist[0][0] * -1 + weightlist[0][1] * train1_x1[item_tr1] + weightlist[0][2] * train1_x2[item_tr1]))
            t_train1_x2.append(sigmoid(
                weightlist[1][0] * -1 + weightlist[1][1] * train1_x1[item_tr1] + weightlist[1][2] * train1_x2[item_tr1]))
        for item_tr2 in range(len(train2_x1)):
            t_train2_x1.append(sigmoid(
                weightlist[0][0] * -1 + weightlist[0][1] * train2_x1[item_tr2] + weightlist[0][2] * train2_x2[item_tr2]))
            t_train2_x2.append(sigmoid(
                weightlist[1][0] * -1 + weightlist[1][1] * train2_x1[item_tr2] + weightlist[1][2] * train2_x2[item_tr2]))
        if case == 1:
            for item_te1 in range(len(test1_x1)):
                t_test1_x1.append(sigmoid(
                    weightlist[0][0] * -1 + weightlist[0][1] * test1_x1[item_te1] + weightlist[0][2] * test1_x2[item_te1]))
                t_test1_x2.append(sigmoid(
                    weightlist[1][0] * -1 + weightlist[1][1] * test1_x1[item_te1] + weightlist[1][2] * test1_x2[item_te1]))
        elif case == 2:
            for item_te1 in range(len(test1_x1)):
                t_test1_x1.append(sigmoid(
                    weightlist[0][0] * -1 + weightlist[0][1] * test1_x1[item_te1] + weightlist[0][2] * test1_x2[item_te1]))
                t_test1_x2.append(sigmoid(
                    weightlist[1][0] * -1 + weightlist[1][1] * test1_x1[item_te1] + weightlist[1][2] * test1_x2[item_te1]))
            for item_te2 in range(len(test2_x1)):
                t_test2_x1.append(sigmoid(
                    weightlist[0][0] * -1 + weightlist[0][1] * test2_x1[item_te2] + weightlist[0][2] * test2_x2[item_te2]))
                t_test2_x2.append(sigmoid(
                    weightlist[1][0] * -1 + weightlist[1][1] * test2_x1[item_te2] + weightlist[1][2] * test2_x2[item_te2]))

    display_output_weight = tk.Label(
        window, width=35, height=2, text=weightlist[-1], fg='black', anchor='w', wraplength=250)
    display_output_weight.place(x=30, y=200)

    # draw
    f = Figure(figsize=(3, 6), dpi=150)
    f.subplots_adjust(left=0.25, right=0.95, top=0.90, bottom=0.1)
    f_plot = f.add_subplot(211)
    f_plot2 = f.add_subplot(212)
    f_plot.tick_params(labelsize=8)
    f_plot2.tick_params(labelsize=8)
    canvas = FigureCanvasTkAgg(f, window)
    canvas.get_tk_widget().place(x=300, y=0)

    # original
    f_plot.clear()
    legend_train1 = f_plot.scatter(
        train1_x1, train1_x2, c='r', marker='o', s=8)
    legend_train2 = f_plot.scatter(
        train2_x1, train2_x2, c='r', marker='x', s=8)
    if case == 1:
        if test_y[0] == 0:
            legend_test1 = f_plot.scatter(
                test1_x1, test1_x2, c='b', marker='o', s=8)
            f.legend([legend_train1, legend_train2, legend_test1],
                     ["train:-1", "train:1", "test:-1"], loc='upper left', prop={'size': 5})
        elif test_y[0] == 1:
            legend_test2 = f_plot.scatter(
                test1_x1, test1_x2, c='b', marker='x', s=8)
            f.legend([legend_train1, legend_train2, legend_test2],
                     ["train:-1", "train:1", "test:1"], loc='upper left', prop={'size': 5})
    elif case == 2:
        legend_test1 = f_plot.scatter(
            test1_x1, test1_x2, c='b', marker='o', s=8)
        legend_test2 = f_plot.scatter(
            test2_x1, test2_x2, c='b', marker='x', s=8)
        f.legend([legend_train1, legend_train2, legend_test1, legend_test2],
                 ["train:-1", "train:1", "test:-1", "test:1"], loc='upper left', prop={'size': 5})
    f_plot.set_xlabel('X1', fontsize=8)
    f_plot.set_ylabel('x2', fontsize=8)

    # transformation
    if nnumber == 2:
        t_x2 = t_train1_x2 + t_train2_x2 + t_test1_x2 + t_test2_x2
        y = []
        y.append(max(t_x2))
        y.append(min(t_x2))
        y = np.array(y)
        if weightlist[-1][1] != 0:
            x = (-1 * weightlist[-1][2] * y +
                weightlist[-1][0]) / weightlist[-1][1]
        f_plot2.clear()
        f_plot2.plot(x, y)
        f_plot2.scatter(t_train1_x1, t_train1_x2, c='r', marker='o', s=8)
        f_plot2.scatter(t_train2_x1, t_train2_x2, c='r', marker='x', s=8)
        if case == 1:
            if test_y[0] == 0:
                f_plot2.scatter(t_test1_x1, t_test1_x2, c='b', marker='o', s=8)
            elif test_y[0] == 1:
                f_plot2.scatter(t_test1_x1, t_test1_x2, c='b', marker='x', s=8)
        elif case == 2:
            f_plot2.scatter(t_test1_x1, t_test1_x2, c='b', marker='o', s=8)
            f_plot2.scatter(t_test2_x1, t_test2_x2, c='b', marker='x', s=8)
        f_plot2.set_xlabel('Y1', fontsize=8)
        f_plot2.set_ylabel('Y2', fontsize=8)

    # colormap
    else:
        totoal_x1 = []
        totoal_x2 = []
        totoal_x1 = list(train_x1) + list(test_x1)
        totoal_x2 = list(train_x2) + list(test_x2)
        h = .01
        x_min, x_max = min(totoal_x1) - .5, max(totoal_x1) + .5
        y_min, y_max = min(totoal_x2) - .5, max(totoal_x2) + .5
        X, Y = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
        outputlist = []
        for w_no in range(len(weightlist)-1):
            outputlist.append(sigmoid(
                weightlist[w_no][0] * -1 + weightlist[w_no][1] * X.ravel() + weightlist[w_no][2] * Y.ravel()))
        finaloutput = weightlist[-1][0] * -1
        for o_no in range(0, len(outputlist)):
            finaloutput += weightlist[-1][o_no + 1] * outputlist[o_no]
        finaloutput = sigmoid(finaloutput)
        finaloutput = finaloutput.reshape(X.shape)
        f_plot2.contourf(X, Y, finaloutput, 150,
                        cmap=cm.get_cmap("PRGn"), alpha=0.5)

        legend_train1 = f_plot2.scatter(
            train1_x1, train1_x2, c='r', marker='o', s=8)
        legend_train2 = f_plot2.scatter(
            train2_x1, train2_x2, c='r', marker='x', s=8)
        if case == 1:
            if test_y[0] == 0:
                legend_test1 = f_plot2.scatter(
                    test1_x1, test1_x2, c='b', marker='o', s=8)
                f.legend([legend_train1, legend_train2, legend_test1],
                        ["train:-1", "train:1", "test:-1"], loc='upper left', prop={'size': 5})
            elif test_y[0] == 1:
                legend_test2 = f_plot2.scatter(
                    test1_x1, test1_x2, c='b', marker='x', s=8)
                f.legend([legend_train1, legend_train2, legend_test2],
                        ["train:-1", "train:1", "test:1"], loc='upper left', prop={'size': 5})
        elif case == 2:
            legend_test1 = f_plot2.scatter(
                test1_x1, test1_x2, c='b', marker='o', s=8)
            legend_test2 = f_plot2.scatter(
                test2_x1, test2_x2, c='b', marker='x', s=8)
            f.legend([legend_train1, legend_train2, legend_test1, legend_test2],
                    ["train:-1", "train:1", "test:-1", "test:1"], loc='upper left', prop={'size': 5})
        f_plot2.set_xlabel('X1', fontsize=8)
        f_plot2.set_ylabel('x2', fontsize=8)

    canvas.draw()
    text.set("Done!!")
    accuracy(weightlist, train_x1, train_x2, train_y, test_x1, test_x2, test_y)


def accuracy(weightlist, train_x1, train_x2, train_y, test_x1, test_x2, test_y):
    train_true = 0
    train_rmse = 0
    for i in range(len(train_y)):
        outputlist = []
        for w_no in range(len(weightlist)-1):
            outputlist.append(sigmoid(
                weightlist[w_no][0] * -1 + weightlist[w_no][1] * train_x1[i] + weightlist[w_no][2] * train_x2[i]))

        finaloutput = weightlist[-1][0] * -1
        for o_no in range(0, len(outputlist)):
            finaloutput += weightlist[-1][o_no + 1] * outputlist[o_no]
        finaloutput = sigmoid(finaloutput)

        if finaloutput > 0.5:
            output = 1
        else:
            output = 0

        if output == train_y[i]:
            train_true += 1
        train_rmse += (output - train_y[i])**2
    train_acc = round(train_true*100 / len(train_y), 1)
    train_rmse = round((math.sqrt(train_rmse / len(train_y))), 2)
    display_train_acc = tk.Label(
        window,  width=10, text=str(train_acc)+"%", fg='black')
    display_train_acc.place(x=120, y=250)
    display_train_rmse = tk.Label(
        window, width=10, text=str(train_rmse), fg='black')
    display_train_rmse.place(x=120, y=310)

    test_true = 0
    test_rmse = 0
    for i in range(len(test_y)):
        outputlist = []
        for w_no in range(len(weightlist)-1):
            outputlist.append(sigmoid(
                weightlist[w_no][0] * -1 + weightlist[w_no][1] * test_x1[i] + weightlist[w_no][2] * test_x2[i]))

        finaloutput = weightlist[-1][0] * -1
        for o_no in range(0, len(outputlist)):
            finaloutput += weightlist[-1][o_no + 1] * outputlist[o_no]
        finaloutput = sigmoid(finaloutput)

        if finaloutput > 0.5:
            output = 1
        else:
            output = 0

        if output == test_y[i]:
            test_true += 1
        test_rmse += (output - test_y[i]) ** 2
    test_acc = round(test_true * 100 / len(test_y), 1)
    test_rmse = round((math.sqrt(test_rmse / len(test_y))), 2)
    display_test_acc = tk.Label(
        window, width=10, text=str(test_acc) + "%", fg='black')
    display_test_acc.place(x=120, y=280)
    display_test_rmse = tk.Label(
        window, width=10, text=str(test_rmse), fg='black')
    display_test_rmse.place(x=120, y=340)


# gui
window = tk.Tk()
window.title("NN HW2-109522050")
window.geometry('750x900')

# select dataset
label_data = tk.Label(window, text="select dataset", fg='black')
label_data.place(x=30, y=10)
button_data = tk.Button(window, text='select', fg='blue', command=openfile)
button_data.place(x=120, y=10)

# input lr
label_lr = tk.Label(window, text="learning rate", fg='black')
label_lr.place(x=30, y=60)
input_lr = tk.DoubleVar()
input_lr.set(0.5)
entry_lr = tk.Entry(window, width=20, textvariable=input_lr)
entry_lr.place(x=120, y=60)

# input epoch
label_epoch = tk.Label(window, text="epoch", fg='black')
label_epoch.place(x=30, y=90)
input_epoch = tk.IntVar()
input_epoch.set(1)
entry_epoch = tk.Entry(window, width=20, textvariable=input_epoch)
entry_epoch.place(x=120, y=90)

# input hidden layer neural number
label_nnumber = tk.Label(window, text="neural number", fg='black')
label_nnumber.place(x=30, y=120)
input_nnumber = tk.IntVar()
input_nnumber.set(2)
entry_nnumber = tk.Entry(window, width=20, textvariable=input_nnumber)
entry_nnumber.place(x=120, y=120)

# display weight
label_output_weight = tk.Label(window, text="output_weight", fg='blue')
label_output_weight.place(x=30, y=170)

# display accuracy
label_train_acc = tk.Label(window, text="training_acc", fg='blue')
label_train_acc.place(x=30, y=250)
label_test_acc = tk.Label(window, text="testing_acc", fg='blue')
label_test_acc.place(x=30, y=280)
label_train_rmse = tk.Label(window, text="training_rmse", fg='blue')
label_train_rmse.place(x=30, y=310)
label_test_rmse = tk.Label(window, text="testing_rmse", fg='blue')
label_test_rmse.place(x=30, y=340)

# display progress bar
progress_bar = ttk.Progressbar(
    window, orient='horizontal', length=250, mode='determinate')
progress_bar.place(x=30, y=400)

# submit botton
button = tk.Button(window, height=2, width=8, text='run',
                   fg='blue', command=perceptron)
button.place(x=120, y=480)

window.mainloop()
