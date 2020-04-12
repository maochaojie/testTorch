import numpy as np
import matplotlib.pylab as plt

def sample_index(recall, precision, sample_num = 100):
    index_list = np.argsort(np.array(recall))
    if len(index_list) < sample_num:
        return recall, precision
    recall = recall[::len(index_list)//sample_num]
    precision = precision[::len(index_list)//sample_num]
    return recall, precision

def plot_curve(name_list, recall_list, precision_list, plot_path, sample_num = 100):
    colorList = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'g', 'b']
    lineList = ['*', '^', 's', 'p', '*', 'v', 'd', 's', '^']
    plt.ioff()
    fig = plt.figure(figsize=(12, 10), dpi = 180)
    for idx, name in enumerate(name_list):
        color_index = idx % (len(colorList))
        line_index = idx // (len(colorList)) % len(lineList)
        style = '{}{}-'.format(colorList[color_index], lineList[line_index])
        recall = recall_list[idx]
        precision = precision_list[idx]
        recall, precision = sample_index(recall, precision, sample_num = sample_num)
        plt.plot(recall, precision, style, label = '{}'.format(name))
    plt.xlabel('time')
    plt.ylabel('price')
    plt.xticks(np.arange(0, max(recall), max(recall)/10))
    plt.yticks(np.arange(0, max(precision), max(precision)/10))
    plt.grid()
    plt.legend(loc=4)
    plt.savefig(plot_path)
    plt.clf()
    plt.cla()
    plt.close()