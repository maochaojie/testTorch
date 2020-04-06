import numpy as np
import torch
class VecDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, pre_length = 7):
        self.all_data = []
        self.file_path = file_path
        self.pre_length = pre_length
        self.collect_data()

    def collect_data(self):
        price_file = self.file_path
        fp = open(price_file, 'r')
        lines = fp.read()
        fp.close()

        index_time = 0
        price_list = []
        for line in lines.split('\n'):
            if line == '':
                continue
            price = line.split('#:#')[0]
            price_list.append(float(price))
            index_time += 1
        all_price_vec = np.array(price_list)
        for idx in range(len(price_list)-1):
            price_vec = np.zeros([self.pre_length], dtype = np.float32)
            if idx < self.pre_length:
                price_vec[self.pre_length - idx:] = all_price_vec[0:idx]
            else:
                price_vec[...] = all_price_vec[idx - self.pre_length:idx]
            predict_val = all_price_vec[idx]
            self.all_data.append([price_vec, predict_val])

    def __getitem__(self, index):
        feature, predict_val =  self.all_data[index]
        return torch.from_numpy(feature), predict_val/10000

    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':
    v = VecDataset('../resource/price_file.csv')
    print(v.all_data)