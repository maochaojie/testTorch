import sys

from tools.tool import plot_curve

price_file = './resource/price_file.csv'

fp = open(price_file, 'r')
lines = fp.read()
fp.close()

index_time = 0
price_list = []
time_list = []
for line in lines.split('\n'):
    if line == '':
        continue
    price = line.split('#:#')[0]
    price_list.append(float(price))
    time_list.append(index_time)
    index_time += 1

name_list = ['raw_price_list']
recall_list = [time_list]
precision_list = [price_list]
plot_path = 'resource/raw_price_view'
sample_num = 100
plot_curve(name_list, recall_list, precision_list, plot_path, sample_num = len(recall_list[0]))