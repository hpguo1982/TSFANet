import torch
import torchvision.models as models


class ResNet50_Features(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNet50_Features, self).__init__()
        self.stem = torch.nn.Sequential(
            original_model.conv1,    # Stage 1: Conv1 (7x7)
            original_model.bn1,
            original_model.relu,
            original_model.maxpool  # Stage 1: MaxPool
        )
        self.layer1 = original_model.layer1  # Stage 2
        self.layer2 = original_model.layer2  # Stage 3
        self.layer3 = original_model.layer3  # Stage 4
        self.layer4 = original_model.layer4  # Stage 5

    def forward(self, x):
        features = []
        x = self.stem(x)     # Stage 1 输出 (1/4 分辨率)
        x = self.layer1(x)   # Stage 2 输出 (1/4 分辨率)
        features.append(x)
        x = self.layer2(x)   # Stage 3 输出 (1/8 分辨率)
        features.append(x)
        x = self.layer3(x)   # Stage 4 输出 (1/16 分辨率)
        features.append(x)
        x = self.layer4(x)   # Stage 5 输出 (1/32 分辨率)
        features.append(x)
        return features  # 返回所有阶段的特征图列表
def get_resnet50_model():
    model = models.resnet34(pretrained=True)
    feature_extractor = ResNet50_Features(model)
    return feature_extractor