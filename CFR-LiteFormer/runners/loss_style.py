from pyexpat import features

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights)
        self.blocks = blocks
        # VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks


        resnet18 = models.resnet18(pretrained=True)
        resnet18.eval()
        # vgg = vgg16_bn(pretrained=True).features
        # vgg.eval()
        self.mean=torch.tensor([0.485, 0.456, 0.406],device='cuda').view(1,3, 1, 1)
        self.std=torch.tensor([0.229, 0.224, 0.225],device='cuda').view(1,3, 1, 1)

        for param in resnet18.parameters():
            param.requires_grad = False

        resnet18 = resnet18.to('cuda')

        self.layer1 = nn.Sequential(*list(resnet18.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet18.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet18.children())[6:7])





    def forward(self, inputs, targets):


        inputs = (inputs - self.mean) / self.std
        targets = (targets - self.mean) / self.std




        input_features =  self.get_features(inputs)

        target_features =  self.get_features(targets)

        loss = 0.0

        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) * w

        return loss
    def get_features(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def perceptual_loss(x, y):
    return F.mse_loss(x, y)


def PerceptualLoss(blocks, weights):
    return FeatureLoss(perceptual_loss, blocks, weights)

def gram_matrix(x):
    b,features=x.size()
    x = x.view(b, -1)
    x = torch.mm(x, x.t()) / features
    return x

def gram_loss(x, y):
    return F.mse_loss(gram_matrix(x), gram_matrix(y))

def TextureLoss(blocks, weights):
    return FeatureLoss(gram_loss, blocks, weights)


# def content_loss(content, pred):
#     return FeatureLoss(perceptual_loss, blocks, weights, device)
#
# def style_loss(style, pred):
#     return FeatureLoss(gram_loss, blocks, weights, device)
#
# def content_style_loss(content, style, pred, alpha, beta):
#     return alpha * content_loss(content, pred) + beta * style_loss(style, pred)
