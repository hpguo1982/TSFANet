import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch
import numpy as np
import torch.nn.functional as F
# from train2d import get_model
import argparse
from torch.utils.data import DataLoader
from dataset import Data



class Evaluator:
    def __init__(self, cuda=True,threshold=0.5):
        self.cuda = cuda
        self.threshold = threshold
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()
        self.IoU = list()


    def evaluate(self, pred, gt):

        pred_binary = (pred >= self.threshold).float().cuda()
        pred_binary_inverse = (pred_binary == 0).float().cuda()
        gt_binary = (gt >= 0.5).float().cuda()
        gt_binary_inverse = (gt_binary == 0).float().cuda()
        # print(pred_binary.shape)
        # print(gt_binary.shape)
        if gt_binary.shape[-1] == 2:
            gt_binary = gt_binary[:,:,0]
        if gt_binary_inverse.shape[-1] == 2:
            gt_binary_inverse = gt_binary_inverse[:,:,0]
        MAE = torch.abs(pred_binary - gt_binary).mean().cuda()
        TP = pred_binary.mul(gt_binary).sum().cuda()
        FP = pred_binary.mul(gt_binary_inverse).sum().cuda()
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda()
        FN = pred_binary_inverse.mul(gt_binary).sum().cuda()
        if TP.item() == 0:
            TP = torch.Tensor([1]).cuda()
        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        Dice = 2 * Precision * Recall / (Precision + Recall)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        IoU = TP / (TP + FP + FN)

        return MAE.data.cpu().numpy().squeeze(), Recall.data.cpu().numpy().squeeze(), Precision.data.cpu().numpy().squeeze(), Accuracy.data.cpu().numpy().squeeze(), Dice.data.cpu().numpy().squeeze(), IoU.data.cpu().numpy().squeeze()


    def update(self, pred, gt):

        mae, recall, precision, accuracy, dice, ioU = self.evaluate(pred, gt)
        self.MAE.append(mae)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)
        self.IoU.append(ioU)

    def show(self ,flag = True):
        if flag == True:
            print("MAE:", "%.2f" % (np.mean(self.MAE ) *100) ,"  Recall:", "%.2f" % (np.mean(self.Recall ) *100), "  Pre:", "%.2f" % (np.mean(self.Precision ) *100),
                  "  Acc:", "%.2f" % (np.mean(self.Accuracy ) *100) ,"  Dice:", "%.2f" % (np.mean(self.Dice ) *100)
                  ,"  IoU:" , "%.2f" % (np.mean(self.IoU ) *100))
            print('\n')
        return np.mean(self.MAE ) *100 ,np.mean(self.Recall ) *100 ,np.mean(self.Precision ) *100 ,np.mean \
            (self.Accuracy ) *100 ,np.mean(self.Dice ) *100 ,np.mean(self.IoU) *100

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt >= 0.5).astype(np.uint8)
            pred_img = (pred >= thresh).astype(np.uint8)
            TP = np.sum((pred_img == 1) & (gt_img == 1))
            TN = np.sum((pred_img == 0) & (gt_img == 0))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))
            if (FN + FP + TP) == 0:
                iou = 0
            else:
                iou_1 = TP / (FN + FP + TP)
                iou_0 = TN / (FN + FP + TN)
                iou = (iou_1 + iou_0) / 2
            iou_list.append(iou)
        final_iou.append(np.mean(iou_list))
    return np.max(final_iou)


def Eval(dataloader_test, model, args2):

    model.eval()
    pred_list = []
    gt_list = []

    if args2.dataset in ['Crack500', 'CrackMap', 'GAPS384', 'TUT']:
        evaluator = Evaluator(threshold=0.5)

    with torch.no_grad():
        for i, sample in enumerate(dataloader_test):
            image, label = sample['image'], sample['label']
            image, label = image.cuda(), label.cuda()
            if label.shape[-1] == 4:
                label = label[:,:,:,:,0]
            label = label.long().squeeze(1)
            logit = model(image, label, False)


            if args2.dataset in ['Crack500', 'CrackMap', 'GAPS384', 'TUT']:


                predictions = torch.argmax(logit, dim=1)
                predictions = F.one_hot(predictions.long(), num_classes=args2.nclass)
                new_labels = F.one_hot(label.long(), num_classes=args2.nclass)
                evaluator.update(predictions[0, :, :, 1], new_labels[0, :, :, 1].float())

                # 添加保存用于 mIoU 的数据（只保存前景通道）
                pred_arr = predictions[0, :, :, 1].float().cpu().numpy()
                gt_arr = new_labels[0, :, :, 1].float().cpu().numpy()
                pred_list.append(pred_arr)
                gt_list.append(gt_arr)

                # 保存可视化图像
                save_dir = '/home/weimengru/dataset/CrackMap/data/epovis_results'
                os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
                idx = i  # 当前样本索引

                # 原图（灰度图或 RGB）
                input_img = image[0].detach().cpu()
                if input_img.shape[0] == 1:  # 单通道
                    input_img = input_img.squeeze(0)
                else:
                    input_img = input_img.permute(1, 2, 0)

                # 预测图（mask）
                pred_mask = predictions[0, :, :, 1].float().cpu().numpy()
                gt_mask = new_labels[0, :, :, 1].float().cpu().numpy()

                # 分别保存三张图
                plt.imsave(f'{save_dir}/{idx:04d}_input.png', input_img.numpy(), cmap='gray')
                plt.imsave(f'{save_dir}/{idx:04d}_pred.png', pred_mask, cmap='gray')
                plt.imsave(f'{save_dir}/{idx:04d}_gt.png', gt_mask, cmap='gray')


    if args2.dataset in ['Crack500', 'CrackMap', 'GAPS384', 'TUT']:
        MAE, Rec, Pre, Acc, Dice, IoU = evaluator.show(False)
        print("MAE: ", "%.2f" % MAE, "  Recall: ", "%.2f" % Rec, " Pre: ", "%.2f" % Pre,
              " Acc: ", "%.2f" % Acc, " F1-score: ", "%.2f" % Dice, " IoU: ", "%.2f" % IoU)
        mIoU = cal_mIoU_metrics(pred_list, gt_list)
        print(f"[mIoU]: {mIoU:.4f}")

    return Dice

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--crop_size", type=int, nargs='+', default=[256, 256], help='H, W')
    parser.add_argument("--nclass", type=int, default=2)
    args2 = parser.parse_args()

    return args2


def test():
    args2 = parse_args()
    model = get_model(args2)
    pre_dict = torch.load('/home/weimengru/dataset/TUT/927weight.pkl', map_location='cpu')
    model.load_state_dict(pre_dict)

    data_test = Data(train=False, dataset=args2.dataset, crop_szie=args2.crop_size)
    dataloader_test = DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=None)
    Eval(dataloader_test, model, args2)


if __name__ == '__main__':
    test()