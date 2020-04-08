import numpy as np
from torch import nn

from tools.tool import plot_curve
from torch_data import vecDataList
import torch
import sys

class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        # self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(30, 1)
    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        out = self.fc2(out)
        return out

class LSTMLR(nn.Module):
    def __init__(self, input_size):
        super(LSTMLR, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, num_layers = 2)
        # self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        ls, out_ch = self.lstm(x)
        #ls = torch.reshape(ls.mean(dim = 1), [-1, 32])
        #ls = torch.reshape(ls[-1,:,:], [-1, 32])
        #torch.gather(ls, )
        # out = self.relu(out)
        out = self.fc2(ls[-1, :, :])
        return out
class Sequence(nn.Module):
    def __init__(self, feature_size = 64):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, feature_size)
        self.lstm2 = nn.LSTMCell(feature_size, feature_size)
        self.linear = nn.Linear(feature_size, 1)
        self.feature_size = feature_size
    def forward(self, input, future = 1):
        future = future - 1
        h_t = torch.zeros(input.size(0), self.feature_size, dtype= torch.float32)
        c_t = torch.zeros(input.size(0), self.feature_size, dtype=torch.float32)
        h_t2 = torch.zeros(input.size(0), self.feature_size, dtype=torch.float32)
        c_t2 = torch.zeros(input.size(0), self.feature_size, dtype=torch.float32)
        outputs = []
        for i, input_t in enumerate(input.chunk(input.size(1), dim = 1)):
            h_t , c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


###
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.ce = nn.MSELoss(size_average = False, reduce = False)
    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = torch.sum(loss, dim = -1)
        return loss.mean()

pre_length = 14
# 预测未来多少天
future = 5
if __name__ == '__main__':
    file_path = './resource/price_file.csv'
    stage = 'test'
    num_workers = 2
    learning_rate = 0.1
    weight_decay = 0.0005
    all_epoches = 50
    save_step = 20
    upload_step = 50
    batch_size = 20


    if stage == 'test':
        batch_size = 2

    dataClass = vecDataList.VecDataset(file_path, stage = stage, pre_length = pre_length, future=future, test_num=100)
    dataLoader = torch.utils.data.DataLoader(dataset = dataClass,
                                             batch_size = batch_size,
                                             shuffle = (stage == 'train'),
                                             num_workers = num_workers,
                                             pin_memory = True,
                                             drop_last = True)

    model = Sequence()
    criterion = MSE()
    params_list = [{"params": model.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr = learning_rate,  weight_decay=weight_decay)

    if stage == 'train':
        model.train()
        eiters = 0
        for epoch in range(all_epoches):
            for i, train_data in enumerate(dataLoader):
                feature, target = train_data
                out = model(feature)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                eiters += 1
                if eiters%10 == 0:
                    print("epoch:{} eiter: {} loss: {}".format(epoch, eiters, loss.item()))
            if (epoch+1) % upload_step == 0 :
                lr = learning_rate * (0.1 ** ((epoch+1)//10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if epoch % save_step == 0:
                state = {'model':model,
                         'model_state':model.state_dict(),
                         'optimizer':optimizer.state_dict()}
                prefix_path = './resource/{}_checkpoint.pth.tar'.format(epoch)
                torch.save(state, prefix_path)
        torch.save(state, prefix_path)
    else:
        epoch = 40
        all_loss = 0
        out_list = []
        target_list = []
        prefix_path = './resource/{}_checkpoint.pth.tar'.format(epoch)
        checkpoint = torch.load(prefix_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        for i, train_data in enumerate(dataLoader):
            feature, target = train_data
            out = model(feature)
            loss = criterion(out[:,-future], target[:, -future])
            all_loss += loss.item()
            out = out[:, -future] * dataClass.std + dataClass.mean
            target = target[:, -future] * dataClass.std + dataClass.mean
            target = target.detach().cpu().numpy()
            output = out.detach().cpu().numpy()
            for index in range(target.shape[0]):
                target_list.append(target[index])
                out_list.append(output[index])
                print("target: {} output:{}".format(target[index], output[index]))
            name_list =['target', 'predict']
            recall_list = [[v+1 for v in range(len(target_list))], [v+1 for v in range(len(target_list))]]
            precision_list = [target_list, out_list]
            plot_path = './resource/price_predict.png'
            plot_curve(name_list, recall_list, precision_list, plot_path, sample_num=len(target_list))
        # up or down
        next_target_list = target_list[1:]
        target_list = target_list[:-1]

        next_out_list = out_list[1:]
        out_list = out_list[:-1]

        target_is_up_down = np.array(next_target_list) - np.array(target_list)
        out_is_up_down = np.array(next_out_list) - np.array(out_list)

        target_is_up_down = target_is_up_down > 0
        out_is_up_down = out_is_up_down > 0

        right_num = np.sum(target_is_up_down * out_is_up_down) + np.sum((1 - target_is_up_down) * (1 - out_is_up_down))
        print('up down precision: {} loss: {}'.format(right_num/target_is_up_down.shape[-1], all_loss/target_is_up_down.shape[-1]))

