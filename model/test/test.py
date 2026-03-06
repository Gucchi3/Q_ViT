
import torch
import torch.nn as nn
import pretty_errors
# from .module import (
#     QuantAct, QuantBNConv2d, QuantLinear, QuantConv2d,
#     DropPath, QStem, QMlp, QFFN,
#     QAttention4D, QAttnFFN, QEmbedding, QEmbeddingAttn,
# )


# def test_cnn(pretrained: bool = False,
#              num_classes: int = 10,
#              in_chans: int = 3,
#              drop_rate: float = 0.0,
#              drop_path_rate: float = 0.0,
#              weight_bit: int = 8,
#              act_bit: int = 8,
#              **kwargs) -> 'test':

#     model = test(num_classes=num_classes, weight_bit=weight_bit, act_bit=act_bit)
#     return model


# class test(nn.Module):
#     def __init__(self):
#         super(test, self).__init__()

#         self.conv0 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
#         self.bn0   = nn.BatchNorm2d(32)
#         self.relu0 = nn.ReLU(inplace=True)
#         self.pool0 = nn.MaxPool2d(2)
        

#         # conv1ブロック
#         self.conv1a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.bn1a = nn.BatchNorm2d(32)
#         self.relu1a = nn.ReLU(inplace=True)
#         self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1b = nn.BatchNorm2d(32)
#         self.relu1b = nn.ReLU(inplace=True)
  




#         # conv2ブロック
#         self.conv2a = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.bn2a = nn.BatchNorm2d(64)
#         self.relu2a = nn.ReLU(inplace=True)
#         self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2b = nn.BatchNorm2d(64)
#         self.relu2b = nn.ReLU(inplace=True)
  

      

#         # conv3ブロック
#         self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3a = nn.BatchNorm2d(128)
#         self.relu3a = nn.ReLU(inplace=True)
#         self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn3b = nn.BatchNorm2d(128)
#         self.relu3b = nn.ReLU(inplace=True)
   




#         self.fc = nn.Linear(128*7*7, 10)  # 1x9
#         #self.fc2 = nn.Linear(5*5*24, 8)

#     def forward(self, x):
#         x = self.conv0(x)
#         x = self.bn0(x)
#         x = self.relu0(x)
        
#         x = self.pool0(x)
        

#         # conv1ブロック
#         x = self.conv1a(x)
#         x = self.bn1a(x)
#         x = self.relu1a(x)
#         x = self.conv1b(x)
#         x = self.bn1b(x)
#         x = self.relu1b(x)



#         # conv2ブロック
#         x = self.conv2a(x)
#         x = self.bn2a(x)
#         x = self.relu2a(x)
#         x = self.conv2b(x)
#         x = self.bn2b(x)
#         x = self.relu2b(x)
       
 

#         # conv3ブロック
#         x = self.conv3a(x)
#         x = self.bn3a(x)
#         x = self.relu3a(x)
#         x = self.conv3b(x)
#         x = self.bn3b(x)
#         x = self.relu3b(x)
  


#         x = torch.flatten(x, 1)
#         out = self.fc(x)
#         return out

class test(nn.Module):
    def __init__(self,
                 pretrained: bool = False,
                 num_classes: int = 10,
                 in_chans: int = 3,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 **kwargs):
        super(test, self).__init__()

        self.conv0 = nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1)
        self.bn0   = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)
        

        # conv1ブロック
        self.conv1a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.bn1a = nn.BatchNorm2d(32)
        self.relu1a = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.bn1b = nn.BatchNorm2d(32)
        self.relu1b = nn.ReLU(inplace=True)
        self.conv1c = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn1c = nn.BatchNorm2d(32)
        self.relu1c = nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.bn1d = nn.BatchNorm2d(32)
        self.relu1d = nn.ReLU(inplace=True)



        # conv2ブロック
        self.conv2a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.bn2a = nn.BatchNorm2d(32)
        self.relu2a = nn.ReLU(inplace=True)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.bn2b = nn.BatchNorm2d(32)
        self.relu2b = nn.ReLU(inplace=True)
        self.conv2c = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn2c = nn.BatchNorm2d(32)
        self.relu2c = nn.ReLU(inplace=True)
        self.conv2d = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.bn2d = nn.BatchNorm2d(32)
        self.relu2d = nn.ReLU(inplace=True)
      

        # conv3ブロック
        self.conv3a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.bn3a = nn.BatchNorm2d(32)
        self.relu3a = nn.ReLU(inplace=True)
        self.conv3b = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.bn3b = nn.BatchNorm2d(64)
        self.relu3b = nn.ReLU(inplace=True)
        self.conv3c = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.bn3c = nn.BatchNorm2d(64)
        self.relu3c = nn.ReLU(inplace=True)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn3d = nn.BatchNorm2d(64)
        self.relu3d = nn.ReLU(inplace=True)

        # conv4ブロック
        self.conv4a = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.bn4a = nn.BatchNorm2d(64)
        self.relu4a = nn.ReLU(inplace=True)
        self.conv4b = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn4b = nn.BatchNorm2d(128)
        self.relu4b = nn.ReLU(inplace=True)
        self.conv4c = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128)
        self.bn4c = nn.BatchNorm2d(128)
        self.relu4c = nn.ReLU(inplace=True)
        self.conv4d = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.bn4d = nn.BatchNorm2d(128)
        self.relu4d = nn.ReLU(inplace=True)
      


        self.fc = nn.Linear(128*7*7, num_classes)
        #self.fc2 = nn.Linear(5*5*24, 8)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        

        # conv1ブロック
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.relu1a(x)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.relu1b(x)
        x = self.conv1c(x)
        x = self.bn1c(x)
        x = self.relu1c(x)
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.relu1d(x)

        # conv2ブロック
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu2b(x)
        x = self.conv2c(x)
        x = self.bn2c(x)
        x = self.relu2c(x)
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.relu2d(x)

        # conv3ブロック
        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)
        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.conv3c(x)
        x = self.bn3c(x)
        x = self.relu3c(x)
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = self.relu3d(x)

        # conv4ブロック
        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu4a(x)
        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.conv4c(x)
        x = self.bn4c(x)
        x = self.relu4c(x)
        x = self.conv4d(x)
        x = self.bn4d(x)
        x = self.relu4d(x)

        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

# class test(nn.Module):
#     def __init__(self):
#         super(test, self).__init__()

#         self.conv0 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
#         self.bn0   = nn.BatchNorm2d(16)
#         self.relu0 = nn.ReLU(inplace=True)
        

#         # conv1ブロック
#         self.conv1a = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
#         self.bn1a = nn.BatchNorm2d(16)
#         self.relu1a = nn.ReLU(inplace=True)
#         self.conv1b = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
#         self.bn1b = nn.BatchNorm2d(16)
#         self.relu1b = nn.ReLU(inplace=True)
#         self.conv1c = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16)
#         self.bn1c = nn.BatchNorm2d(16)
#         self.relu1c = nn.ReLU(inplace=True)
#         self.conv1d = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
#         self.bn1d = nn.BatchNorm2d(16)
#         self.relu1d = nn.ReLU(inplace=True)



#         # conv2ブロック
#         self.conv2a = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
#         self.bn2a = nn.BatchNorm2d(16)
#         self.relu2a = nn.ReLU(inplace=True)
#         self.conv2b = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
#         self.bn2b = nn.BatchNorm2d(32)
#         self.relu2b = nn.ReLU(inplace=True)
#         self.conv2c = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
#         self.bn2c = nn.BatchNorm2d(32)
#         self.relu2c = nn.ReLU(inplace=True)
#         self.conv2d = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
#         self.bn2d = nn.BatchNorm2d(32)
#         self.relu2d = nn.ReLU(inplace=True)
      

#         # conv3ブロック
#         self.conv3a = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
#         self.bn3a = nn.BatchNorm2d(32)
#         self.relu3a = nn.ReLU(inplace=True)
#         self.conv3b = nn.Conv2d(32, 48, kernel_size=1, stride=1, padding=0)
#         self.bn3b = nn.BatchNorm2d(48)
#         self.relu3b = nn.ReLU(inplace=True)
#         self.conv3c = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, groups=48)
#         self.bn3c = nn.BatchNorm2d(48)
#         self.relu3c = nn.ReLU(inplace=True)
#         self.conv3d = nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0)
#         self.bn3d = nn.BatchNorm2d(48)
#         self.relu3d = nn.ReLU(inplace=True)

#         # conv4ブロック
#         self.conv4a = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, groups=48)
#         self.bn4a = nn.BatchNorm2d(48)
#         self.relu4a = nn.ReLU(inplace=True)
#         self.conv4b = nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0)
#         self.bn4b = nn.BatchNorm2d(64)
#         self.relu4b = nn.ReLU(inplace=True)
#         self.conv4c = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
#         self.bn4c = nn.BatchNorm2d(64)
#         self.relu4c = nn.ReLU(inplace=True)
#         self.conv4d = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
#         self.bn4d = nn.BatchNorm2d(64)
#         self.relu4d = nn.ReLU(inplace=True)
      


#         self.fc = nn.Linear(64*5*5, 8)  # 1x9
#         #self.fc2 = nn.Linear(5*5*24, 8)

#     def forward(self, x):
#         x = self.conv0(x)
#         x = self.bn0(x)
#         x = self.relu0(x)
        

#         # conv1ブロック
#         x = self.conv1a(x)
#         x = self.bn1a(x)
#         x = self.relu1a(x)
#         x = self.conv1b(x)
#         x = self.bn1b(x)
#         x = self.relu1b(x)
#         x = self.conv1c(x)
#         x = self.bn1c(x)
#         x = self.relu1c(x)
#         x = self.conv1d(x)
#         x = self.bn1d(x)
#         x = self.relu1d(x)

#         # conv2ブロック
#         x = self.conv2a(x)
#         x = self.bn2a(x)
#         x = self.relu2a(x)
#         x = self.conv2b(x)
#         x = self.bn2b(x)
#         x = self.relu2b(x)
#         x = self.conv2c(x)
#         x = self.bn2c(x)
#         x = self.relu2c(x)
#         x = self.conv2d(x)
#         x = self.bn2d(x)
#         x = self.relu2d(x)

#         # conv3ブロック
#         x = self.conv3a(x)
#         x = self.bn3a(x)
#         x = self.relu3a(x)
#         x = self.conv3b(x)
#         x = self.bn3b(x)
#         x = self.relu3b(x)
#         x = self.conv3c(x)
#         x = self.bn3c(x)
#         x = self.relu3c(x)
#         x = self.conv3d(x)
#         x = self.bn3d(x)
#         x = self.relu3d(x)

#         # conv4ブロック
#         x = self.conv4a(x)
#         x = self.bn4a(x)
#         x = self.relu4a(x)
#         x = self.conv4b(x)
#         x = self.bn4b(x)
#         x = self.relu4b(x)
#         x = self.conv4c(x)
#         x = self.bn4c(x)
#         x = self.relu4c(x)
#         x = self.conv4d(x)
#         x = self.bn4d(x)
#         x = self.relu4d(x)

#         x = torch.flatten(x, 1)
#         out = self.fc(x)
#         return out
    




# ── 動作確認 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchinfo import summary

    model = test()
    summary(model, input_size=(1, 3, 224, 224))