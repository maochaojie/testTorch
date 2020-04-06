import numpy as np
from torch import nn
from torch_data import vecDataList
import torch

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
###
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.ce = nn.MSELoss()
    def forward(self, input, target):
        loss = self.ce(input, target)
        return loss

if __name__ == '__main__':
    file_path = './resource/price_file.csv'
    pre_length = 14
    stage = 'train'
    num_workers = 2
    learning_rate = 0.1
    mom = 0.9
    weight_decay = 0.0005
    all_epoches = 50
    batch_size = 100
    dataClass = vecDataList.VecDataset(file_path, pre_length = pre_length)
    dataLoader = torch.utils.data.DataLoader(dataset = dataClass,
                                             batch_size = batch_size,
                                             shuffle = (stage == 'train'),
                                             num_workers = num_workers,
                                             pin_memory = True,
                                             drop_last = True)

    model = LSTMLR(1)
    criterion = MSE()
    params_list = [{"params": model.parameters()}]
    optimizer = torch.optim.SGD(params_list, lr = learning_rate, momentum=mom, weight_decay=weight_decay)

    stage = 'train'
    if stage == 'train':
        model.train()

        eiters = 0
        for epoch in range(all_epoches):
            for i, train_data in enumerate(dataLoader):
                feature, target = train_data
                feature = torch.reshape(feature, [-1, pre_length, 1]).permute(1, 0, 2)
                target = torch.reshape(target, [-1, 1])
                out = model(feature)
                loss = criterion(out, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                eiters += 1
                if eiters%2 == 0:
                    print("epoch:{} eiter: {} loss: {}".format(epoch, eiters, loss.item()))
            if (epoch+1) % 10 == 0 :
                lr = learning_rate * (0.1 ** ((epoch+1)//10))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if epoch % 5 == 0:
                state = {'model':model,
                         'model_state':model.state_dict(),
                         'optimizer':optimizer.state_dict()}
                prefix_path = './resource/{}_checkpoint.pth.tar'.format(epoch)
                torch.save(state, prefix_path)
    else:
        epoch = 5
        prefix_path = './resource/{}_checkpoint.pth.tar'.format(epoch)
        checkpoint = torch.load(prefix_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        for i, train_data in enumerate(dataLoader):
            feature, target = train_data
            feature = torch.reshape(feature, [-1, pre_length, 1])
            target = torch.reshape(target, [-1, 1])
            out = model(feature)
            print(target*10000, out*10000)
