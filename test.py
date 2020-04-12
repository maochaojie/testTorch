import torch
import numpy as np

from tools.tool import plot_curve
from torch_data import vecDataList
import train
from train import Sequence, MSE
import json

if __name__ == '__main__':
    file_path = './resource/price_file_test.csv'
    fp = open(file_path, 'r')
    lines = fp.read()
    fp.close()
    data_list = []
    for line in lines.split('\n'):
        if line == '':
            continue
        data_list.append(float(line))

    pre_length = train.pre_length
    # 预测未来多少天
    future = 7


    epoch = 40
    model = Sequence()
    out_list = []
    prefix_path = './resource/{}_checkpoint.pth.tar'.format(epoch)
    checkpoint = torch.load(prefix_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])

    json_str = json.load(open('./resource/sta.json','r'))
    std = json_str['std']
    mean = json_str['mean']

    input_data = (np.array(data_list) - mean)/std
    input_vec = np.zeros(pre_length, dtype = np.float32)
    if len(data_list) < pre_length:
        input_vec[pre_length - len(data_list):] = input_data
    else:
        input_vec[...] = input_data[-pre_length:]
    feature = torch.from_numpy(input_vec).reshape(1, -1)

    out = model(feature, future)
    output = out.detach().cpu().numpy().reshape(-1) * std + mean
    old_value = 0
    for index in range(output.shape[-1]):
        out_list.append(output[index])
        if index >= pre_length:
            if output[index] > old_value:
                up_down = 'up {}'.format(output[index] - old_value)
            else:
                up_down = 'down {}'.format(output[index] - old_value)
            print('next {} days price: {}, up_down: {}'.format(index - pre_length + 1, output[index], up_down))
        old_value = output[index]
    name_list =['pre_days', 'predict_days']

    recall_list = [[v for v in range(len(data_list))], [v + len(data_list) - pre_length for v in range(len(out_list))]]
    precision_list = [data_list, out_list]
    plot_path = './resource/next_price_predict.png'
    plot_curve(name_list, recall_list, precision_list, plot_path, sample_num=len(out_list) + len(data_list))