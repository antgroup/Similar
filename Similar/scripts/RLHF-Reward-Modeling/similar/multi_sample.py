import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

# 参考 https://blog.csdn.net/qq_45270993/article/details/134171370

# ddp需要使用的包
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 设置local_rank,默认设置成-1。因为如果使用torch.distributed.launch启动ddp，则local_rank自动为每张卡分配。从0开始。要记住每张卡的程序的区别就是local_rank不同。因为我们要使用主进程测试和保存模型
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type=int, default=-1)
args = parser.parse_args()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return x

# 初始化ddp环境，为每个进程分配卡
if args.local_rank >= 0:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    ddp = True
else:
    ddp = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./', train=True, download=False, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 这里要注意，多卡训练需要使用ddp的方式设置sample。保证每张卡分配的批次是不同的
if ddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# 测试只需要在一张卡上进行，因为要使用一张卡在所有测试数据上进行测试

test_dataset = datasets.MNIST('./', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

model = SimpleNN().to(device)

# 使用ddp封装模型，同时对学习率进行了放大。因为使用学习率*梯度更新参数。每张卡的模型参数是一致的，由于多卡训练导致有效批次增大。比如之前是单卡学习率0.01，反向传播的时候使用0.01*一个批次的梯度进行参数更新。现在是多卡，所以是0.01*gpu数目*gpu数目的批次的梯度更新参数
if ddp:
    model = DDP(model, device_ids=[args.local_rank])
    gpu_num = torch.distributed.get_world_size()
else:
    gpu_num = 1

optimizer = optim.SGD(model.parameters(), lr=0.01*gpu_num)

# Initializing the loss function outside of the loops
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(5):
    train_sampler.set_epoch(epoch)    # 更新随机种子，可选择使用
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 测试时仅使用主进程
if not ddp or (ddp and dist.get_rank() == 0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n")


# ddp使用主进程保存模型，不然的话保存多次没必要。

if not ddp or (ddp and dist.get_rank() == 0):
    torch.save(model.state_dict(), 'ddp_model.pth')
