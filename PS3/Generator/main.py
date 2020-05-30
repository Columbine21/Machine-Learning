import pandas as pd
import heapq

import matplotlib.pyplot as plt
import torch
from poutyne.framework import Model, Callback
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from Generator.dataset import NameDataset, StringToPaddedIndexesWithLength
from Generator.models.recurrent_model import Generator
# from Generator.utils import char_to_idx, PAD_IDX, idx_to_char

from Generator.utils import char_to_idx, idx_to_char, PAD_IDX, EOS


class ClipGradient(Callback):
    def __init__(self, module: Generator, clip):
        super().__init__()
        self.clip = clip
        self.module = module

    def on_batch_end(self, batch, logs):
        torch.nn.utils.clip_grad_value_(self.module.parameters(), self.clip)


class GenerateCallback(Callback):
    def __init__(self, net: Generator, device, n=10, every=10):
        super().__init__()
        self.net = net
        self.device = device
        self.n = n
        self.every = every

    def on_epoch_end(self, epoch, logs):
        if epoch % self.every == 0:
            self.net.train(False)
            print('\n'.join([self.net.generate(self.device) for i in range(self.n)]))
            self.net.train(True)


def visualize(model, device, prefix, max_len=40):
    print("\t\t\tInput prefix is : " + prefix)
    print("\t**********************************************************")
    index = prefix_length = len(prefix)
    selected_char = generated_str = prefix
    transformer = StringToPaddedIndexesWithLength(max_len)
    # 全零初始化 lstm 的状态向量
    hidden_size = (model.lstm_layers, 1, model.hidden_size)
    h_n, c_n = torch.zeros(*hidden_size, device=device), torch.zeros(*hidden_size, device=device)

    while selected_char != EOS and index < max_len:
        x = transformer(selected_char).unsqueeze(0).to(device)
        x, (h_n, c_n) = model._forward(x, (h_n, c_n))
        x = x[len(selected_char)-1]
        probability_list = torch.nn.functional.softmax(x, dim=0).detach().numpy()
        print(x)
        print("\t**********************************************************")
        to_numpy = x.detach().numpy()
        top_five = heapq.nlargest(5, range(len(to_numpy)), to_numpy.take)

        print(f"\t\t\tTop five selection: {[idx_to_char[item] for item in top_five]}")
        print(f"\t\t\tAnd Their Probability: {[probability_list[item] for item in top_five]}")

        distribution = Categorical(logits=x)
        selected_char = idx_to_char[distribution.sample().item()]

        print("Select (sample result): " + selected_char)
        if selected_char == EOS:
            break
        generated_str += selected_char
        index += 1

    return generated_str

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = NameDataset("../data/*male.txt")
    # loader = DataLoader(dataset, batch_size=32, shuffle=True,
    #                     num_workers=4,
    #                     collate_fn=dataset.sort_by_length_flatten_on_timestamp_collate)
    # net = Generator(len(char_to_idx))
    # optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)
    # criterion = CrossEntropyLoss(ignore_index=PAD_IDX)
    # model = Model(net, optimizer, criterion, metrics=['accuracy']).to(device)
    # history = model.fit_generator(loader, epochs=300, validation_steps=0,
    #                               callbacks=[ClipGradient(net, 2), GenerateCallback(net, device)])
    # torch.save(net, "checkpoint/lstm_model.pkl")
    # df = pd.DataFrame(history).set_index('epoch')
    # df.plot(subplots=True)
    # plt.show()
    model = torch.load('checkpoint/lstm_model.pkl')

    print(visualize(model, device, "ja"))
