# TSFANet

## 🔗 Dataset

download the dataset [Crack500](https://github.com/guoguolord/CrackDataset) ,  [CrackMap](https://github.com/ikatsamenis/CrackMap) and [GAPS384](https://www.kaggle.com/datasets/vangiap/gaps384)

## Experimental Environment

```
pip install -r requirements.txt
```

## Usage

### Training

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Crack500 --end_epoch 200 --warm_epochs 5 --lr 0.0003 --train_batchsize 8 --crop_size 512 512 --nclass 2 
```

### Testing

```
 CUDA_VISIBLE DEVIcEs=0 python test.py --dataset Crack500 --crop_size 512 512--nclass 2
```
