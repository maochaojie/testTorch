import numpy as np
import torch
import json
class VecDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, stage, future = 1, pre_length = 7, test_num = 50):
        self.all_data = []
        self.file_path = file_path
        self.pre_length = pre_length
        self.test_num = test_num
        self.future = future
        self.stage = stage
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
        all_price_vec = np.array(price_list[:-self.test_num])
        self.mean = np.mean(all_price_vec)
        self.std = np.std(all_price_vec)
        meta_json = {}
        meta_json['mean'] = self.mean
        meta_json['std'] = self.std
        json_str = json.dumps(meta_json)
        with open('./resource/sta.json', 'w') as f:
            f.write(json_str)
        if self.stage == 'train':
            print("data mean:{}, std:{}".format(self.mean, self.std))
            all_price_vec = (all_price_vec - self.mean) / self.std
            for idx in range(all_price_vec.shape[-1]-self.future):
                price_vec = np.zeros([self.pre_length], dtype = np.float32)
                predict_val = np.zeros([self.pre_length + self.future - 1], dtype = np.float32)
                if idx < self.pre_length:
                    price_vec[self.pre_length - idx:] = all_price_vec[0:idx]
                    predict_val[self.pre_length - idx:] = all_price_vec[1:idx+self.future]
                else:
                    price_vec[...] = all_price_vec[idx - self.pre_length:idx]
                    predict_val[...] = all_price_vec[idx - self.pre_length + 1:idx + self.future]
                self.all_data.append([price_vec, predict_val])
        else:
            print("data mean:{}, std:{}".format(self.mean, self.std))
            all_price_vec = np.array(price_list[-self.test_num-self.pre_length:])
            all_price_vec = (all_price_vec - self.mean) / self.std
            for idx in range(self.pre_length, all_price_vec.shape[-1]- self.future, 1):
                price_vec = np.zeros([self.pre_length], dtype=np.float32)
                predict_val = np.zeros([self.pre_length + self.future - 1], dtype=np.float32)
                if idx < self.pre_length:
                    price_vec[self.pre_length - idx:] = all_price_vec[0:idx]
                    predict_val[self.pre_length - idx:] = all_price_vec[1:idx + self.future]
                else:
                    price_vec[...] = all_price_vec[idx - self.pre_length:idx]
                    predict_val[...] = all_price_vec[idx - self.pre_length + 1:idx + self.future]
                self.all_data.append([price_vec, predict_val])

    def __getitem__(self, index):
        feature, predict_val =  self.all_data[index]
        return torch.from_numpy(feature), predict_val

    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':
    v = VecDataset('../resource/price_file.csv')
    print(v.all_data)