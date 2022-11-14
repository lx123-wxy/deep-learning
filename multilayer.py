import torch
from torch import nn
from d2l import torch as d2l

batch_size = 196
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 196

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens * 6, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens * 6, requires_grad=True))
W2 = nn.Parameter(torch.rand(
    num_hiddens * 6, num_hiddens * 4, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens * 4, requires_grad=True))
W3 = nn.Parameter(torch.rand(
    num_hiddens * 4, num_hiddens * 2, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_hiddens * 2, requires_grad=True))
W4 = nn.Parameter(torch.rand(
    num_hiddens * 2, num_hiddens, requires_grad=True) * 0.01)
b4 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W5 = nn.Parameter(torch.rand(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b5 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    H2 = relu(H1@W2 + b2)
    H3 = relu(H2@W3 + b3)
    return H3@W4 + b4


loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 50, 0.01
updater = torch.optim.Adamax(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()
