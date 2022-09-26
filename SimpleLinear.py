import random
import torch


# 矩阵.shape 当矩阵只有一行是size为列数 其余情况显示行列数
# len(a) 当矩阵a只有一行n列时len表示的是列数,当矩阵为n行m列时表示的是行数
# feature 特征矩阵 labels 标签向量
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # normal(means, std, out=None, size(a, b))
    # means:均值 std:标准差 out:可选的输出张量 size a行b列的矩阵 len(w)表示w矩阵行数
    y = torch.matmul(X, w) + b  # tensor的乘法
    print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # -1代表n n=tensor的长度 这个return的reshape就是调成一列


true_w = torch.tensor([3, -4.8])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 疑问:矩阵相乘X是num行2列, w是1行2列, 结果为1行num列？
# print(torch.matmul(torch.normal(0, 1, (5, 2)), true_w))


# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# requires_grad设置为True,则反向传播时,该tensor就会自动求导
# is_leaf 默认为true 求导（梯度）条件 backward
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 线性回归模型
def Lineag(X, W, B):
    return torch.matmul(X, W) + B


# y_hat预测值 y真实值
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 模型参数集合、学习速率和批量大小
def sgd(params, lr, batch_size):
    """小批量batch_size随机梯度grad下降。"""
    with torch.no_grad():
        # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 当param求过一次梯度时 清除梯度值重新赋值为零


lr = 0.03  # 学习率
num_epochs = 3
net = Lineag
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()  # backward 计算并储存梯度值
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')
