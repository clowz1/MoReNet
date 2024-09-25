import argparse
import os
import time

import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from model.dataset import MoReDataset
from model.model import MoRe

parser = argparse.ArgumentParser(description='MORE')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--load-model', type=str, default='sample.model')
parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default="/home/czc/catkin_ws/src/morenet/2.22",
                    help='pre-trained model path')
parser.add_argument('--data-path', type=str, default="/home/czc/catkin_ws/src/morenet/2.22",
                    help='data path')
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=5)

args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = 1
else:
    args.cuda = 0

if args.cuda:
    torch.cuda.manual_seed(1)

logger = SummaryWriter(os.path.join('./assets/log/', args.tag))  # 写入文件间隔
np.random.seed(int(time.time()))


def worker_init_fn(pid):
    np.random.seed(torch.initial_seed() % (2 ** 31 - 1))


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))  
    return torch.utils.data.dataloader.default_collate(batch)  


input_viewpoint = [0, 1, 2, 3, 4, 5, 6, 7, 8]
input_size = 100
embedding_size = 128
# 特征维度
joint_size = 22
thresh_acc = [0.2, 0.25, 0.3]
joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])

train_loader = torch.utils.data.DataLoader(
    ShadowPairedDataset(
        path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=True,
    ),
    batch_size=args.batch_size,
    num_workers=0,
    pin_memory=True,  # 拷贝数据到cuda
    shuffle=True,
    worker_init_fn=worker_init_fn,  # 初始化每一个worker
    collate_fn=my_collate,  # 自定义取出数据
)

test_loader = torch.utils.data.DataLoader(
    ShadowPairedDataset(
        path=args.data_path,
        input_size=input_size,
        input_viewpoint=input_viewpoint,
        is_train=False,
        with_name=True,
    ),
    batch_size=args.batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    # ！=是≠
    is_resume = 1

# if is_resume or args.mode == 'test':
#     model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
#     model.device_ids = [args.gpu]
#     print('load model {}'.format(args.load_model))
# else:
model = MoRe(input_size=input_size, embedding_size=embedding_size, joint_size=joint_size)

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [1, 2]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
# 优化器
scheduler = StepLR(optimizer, step_size=80, gamma=0.5)


def train(model, loader, epoch):
    # scheduler.step()  # 每过80次更新学习率
    model.train()
    torch.set_grad_enabled(True)
    # train_error_shadow = 0
    train_error_hand = 0
    error_each_angle = []
    correct_hand = [0, 0, 0]
    # 初始化
    for batch_idx, (hand, target) in enumerate(loader):  # 遍历
        torch.cuda.empty_cache()
        if args.cuda:
            hand, target = hand.cuda(), target.cuda()

        optimizer.zero_grad()  # 梯度归零

        embedding_hand, joint_hand = model(hand, is_hand=True)
        loss_hand_reg = F.smooth_l1_loss(joint_hand, target)
        loss_hand_cons = constraints_loss(joint_hand) / target.shape[0]
        loss_hand = loss_hand_reg + loss_hand_cons
        loss = loss_hand
        loss.backward()
        optimizer.step()
        scheduler.step()
        # 更新参数

        loss = loss_hand.cuda()

        # compute acc
        error_each_angle = abs(joint_hand.cpu().data.numpy() - target.cpu().data.numpy())
        list1 = [error_each_angle[0][0], error_each_angle[0][1],error_each_angle[0][2],error_each_angle[0][3],error_each_angle[0][4],error_each_angle[0][5],error_each_angle[0][6],error_each_angle[0][7],error_each_angle[0][8],error_each_angle[0][9],error_each_angle[0][10],error_each_angle[0][11],error_each_angle[0][12],error_each_angle[0][13],error_each_angle[0][14],error_each_angle[0][15],error_each_angle[0][16],error_each_angle[0][17],error_each_angle[0][18],error_each_angle[0][19],error_each_angle[0][20],error_each_angle[0][21]]
        data = pd.DataFrame([list1])
        data.to_csv('./error.csv', mode='a', header=False, index=False)
        res_hand = [np.sum(np.sum(abs(joint_hand.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                                   axis=-1) > 20) for thresh in thresh_acc]
        correct_hand = [c + r for c, r in zip(correct_hand, res_hand)]
        # 打包数据

        # compute average angle error 误差绝对值总和求平均
        train_error_hand += F.smooth_l1_loss(joint_hand, target, size_average=False) / joint_size

        if batch_idx % args.log_interval == 0:
            if isinstance(loss_hand_cons, float):
                loss_hand_cons = torch.zeros(1)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_reg_hand: {:.6f}\t'
                  'Loss_cons_hand: {:.6f}\t{}'.format(
                epoch, batch_idx * args.batch_size, len(loader.dataset),
                       100. * batch_idx * args.batch_size / len(loader.dataset),
                loss.item(), loss_hand_reg.item(), loss_hand_cons.item(), args.tag))

            logger.add_scalar('train_loss', loss.item(),
                              epoch)

            logger.add_scalar('train_loss_hand_reg', loss_hand_reg.item(),
                              batch_idx + epoch * len(loader))
            logger.add_scalar('train_loss_hand_cons', loss_hand_cons.item(),
                              batch_idx + epoch * len(loader))
            logger.add_graph(NewTeachingTeleModel(), hand)

    train_error_hand /= len(loader.dataset)
    acc_hand = [float(c) / float(len(loader.dataset)) for c in correct_hand]

    return acc_hand, train_error_hand, error_each_angle


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)

    test_loss_hand_reg = 0
    test_loss_hand_cons = 0
    test_error_hand = 0
    res = []
    error_each_angle = []
    correct_hand = [0, 0, 0]
    for hand, target, name in loader:
        if args.cuda:
            hand, target = hand.cuda(), target.cuda()

        embedding_hand, joint_hand = model(hand, is_human=True)
        test_loss_hand_reg += F.mse_loss(joint_hand, target, size_average=False).item()
        cons = constraints_loss(joint_hand)
        if not isinstance(cons, float):
            test_loss_hand_cons += cons

        res_hand = [np.sum(np.sum(abs(joint_hand.cpu().data.numpy() - target.cpu().data.numpy()) < thresh,
                                   axis=-1) == joint_size) for thresh in thresh_acc]
        correct_hand = [c + r for c, r in zip(correct_hand, res_hand)]
        test_error_hand += F.l1_loss(joint_hand, target, size_average=False) / joint_size
        res.append((name, joint_hand))



    test_loss_hand_reg /= len(loader.dataset)
    test_loss_hand_cons /= len(loader.dataset)
    test_loss = test_loss_hand_reg + test_loss_hand_cons
    test_error_hand /= len(loader.dataset)

    acc_hand = [float(c) / float(len(loader.dataset)) for c in correct_hand]



    return acc_hand, test_error_hand, test_loss, test_loss_hand_reg, test_loss_hand_cons, error_each_angle


def constraints_loss(joint_angle):
    F4 = [joint_angle[:, 0], joint_angle[:, 5], joint_angle[:, 9], joint_angle[:, 13]]
    F1_3 = [joint_angle[:, 1], joint_angle[:, 6], joint_angle[:, 10], joint_angle[:, 14],
            joint_angle[:, 2], joint_angle[:, 7], joint_angle[:, 11], joint_angle[:, 15],
            joint_angle[:, 3], joint_angle[:, 8], joint_angle[:, 12], joint_angle[:, 16],
            joint_angle[:, 21]]
    loss_cons = 0.0

    for pos in F1_3:
        for f in pos:
            loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.57, 0)
    for pos in F4:
        for f in pos:
            loss_cons = loss_cons + max(-0.349 - f, 0) + max(f - 0.349, 0)
    for f in joint_angle[:, 4]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 0.785, 0)
    for f in joint_angle[:, 17]:
        loss_cons = loss_cons + max(-1.047 - f, 0) + max(f - 1.047, 0)
    for f in joint_angle[:, 18]:
        loss_cons = loss_cons + max(0 - f, 0) + max(f - 1.222, 0)
    for f in joint_angle[:, 19]:
        loss_cons = loss_cons + max(-0.209 - f, 0) + max(f - 0.209, 0)
    for f in joint_angle[:, 20]:
        loss_cons = loss_cons + max(-0.524 - f, 0) + max(f - 0.524, 0)
    return loss_cons


def main():
    if args.mode == 'train':
        for epoch in range(is_resume * args.load_epoch, args.epoch):
            acc_train_hand, train_error_hand, error_each_angle = train(model, train_loader, epoch)
            print('Train done, acc_hand={}, train_error_hand={}'.format(acc_train_hand, train_error_hand))

            logger.add_scalar('train_acc_hand0.2', acc_train_hand[0], epoch)
            logger.add_scalar('train_acc_hand0.25', acc_train_hand[1], epoch)
            logger.add_scalar('train_acc_hand0.3', acc_train_hand[2], epoch)
            if epoch % args.save_interval == 0:
                path = os.path.join(args.model_path, args.tag + '_{}.model'.format(epoch))
                torch.save(model, path)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc_test_hand, test_error_hand, loss, loss_hand_reg, loss_hand_cons, error_each_angle = test(model,
                                                                                                         test_loader)
        print(
            'Test done, acc_hand={}, error_hand ={}, loss={}, loss_hand_reg={}, loss_hand_cons={}'.format(
                acc_test_hand,
                test_error_hand,
                loss, loss_hand_reg,
                loss_hand_cons))


if __name__ == "__main__":
    main()
