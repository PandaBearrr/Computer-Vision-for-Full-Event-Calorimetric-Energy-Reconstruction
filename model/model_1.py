import torch
import torch.nn.functional as F
from torch import nn

'''''''''''''''''''''
该文件中的网络均使用以下学习率调度器:
def lf_function(epoch): 
    if epoch < warmup_epochs_1:
        return 1
    elif epoch < warmup_epochs_2: 
        return 0.1
    elif epoch < warmup_epochs_3:
        return((epoch - warmup_epochs_2) / (warmup_epochs_3 - warmup_epochs_2)) * 0.5 + 0.1
    else:
        return(((1 + math.cos((epoch - warmup_epochs_3) * math.pi / (args.epochs - warmup_epochs_3))) / 2) * 0.5 + 0.1)
'''''''''''''''''''''


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1) #H*W->H*W
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # 生成查询、键、值
        queries = self.query(x).view(batch_size, C, -1) # (B, C, H*W)
        keys = self.key(x).view(batch_size, C, -1) # (B, C, H*W)
        values = self.value(x).view(batch_size, C, -1) # (B, C, H*W)

        # 计算自注意力
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys) # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)

        out = torch.bmm(values, attention_scores.permute(0, 2, 1)) # (B, C, H*W)
        return out.view(batch_size, C, H, W) #不改变形状

class RowSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(RowSelfAttention, self).__init__()
        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x 的形状为 (B, C, H, W)
        B, C, H, W = x.shape
        
        # 将输入重新排列为 (B, H, W, C)，
        # 使得每一行（H 方向）成为一个序列，序列长度为 W，每个“词”是 C 维向量
        x = x.permute(0, 2, 3, 1)
        
        # 对每个序列中的每个位置应用共享的线性层，生成查询、键和值
        # 得到的形状均为 (B, H, W, C)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # 对于每一行序列，计算自注意力得分
        # Q 与 K 的转置相乘，注意：这里每一行内部进行点积计算，
        # 结果形状为 (B, H, W, W)，即每个序列中各位置之间的相关性
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        # （可选）可以添加缩放因子，例如除以 sqrt(C)
        attn_scores = attn_scores / (C ** 0.5)
        
        # 对每行的得分应用 softmax，使每行中注意力权重归一化
        attn_weights = self.softmax(attn_scores)
        
        # 使用注意力权重对值向量加权求和，得到每个位置的新表示
        # 输出形状仍为 (B, H, W, C)
        out = torch.matmul(attn_weights, V)
        
        # 将输出重新排列回 (B, C, H, W) 的形状
        out = out.permute(0, 3, 1, 2)
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock,self).__init__()
        self.fc1=nn.Linear(channels,channels//reduction,bias=False)
        self.fc2=nn.Linear(channels//reduction,channels,bias=False)

    def forward(self,x):
        b, c,_,_=x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c) # Squeeze
        y=F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1) # Excitation - 2nd layer
        return x * y.expand_as(x) # Scale

class DownSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2卷积，步幅2会让特征尺寸减半
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)
#定义上采样层
class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 改变通道数的卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值进行上采样
        x = self.Up(up)  # 改变输出通道数
        x = torch.cat([x, r], dim=1)  # 进行跳跃连接，拼接特征
        return x


class Conv_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv_UNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1),  # 3x3卷积，padding=1保持尺寸不变
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DownSampling_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling_UNet, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2卷积，步幅2会让特征尺寸减半
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)

class UpSampling_UNet(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling_UNet, self).__init__()
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 改变通道数的卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值进行上采样
        x = self.Up(up)  # 改变输出通道数
        x = torch.cat([x, r], dim=1)  # 进行跳跃连接，拼接特征
        return x
        



class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            # nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x


class CNNwithSEBlock_1(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock_1,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.se1=SEBlock(32)
        self.decoder=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            # nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x



class CNNwithRowSelfAttention_1(nn.Module):
    def __init__(self):
        super(CNNwithRowSelfAttention_1,self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.attention = RowSelfAttention(32)
        self.decoder=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            # nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x=self.encoder(x)
        x=self.attention(x)
        x=self.decoder(x)
        return x
    

class CNN3D_1(nn.Module):
    def __init__(self):
        super(CNN3D_1,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(12, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            # nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56)
        x=self.encoder(x)
        x=self.decoder(x)
        return x


class CNNwithSEBlock3D_1(nn.Module):
    def __init__(self):
        super(CNNwithSEBlock3D_1,self).__init__()
        self.conv3x3x3 = nn.Conv3d(1, 2, kernel_size=3, padding=(0,1,1))
        self.conv3x5x5 = nn.Conv3d(1, 2, kernel_size=(3,5,5), padding=(0,2,2))
        self.conv3x7x7 = nn.Conv3d(1, 2, kernel_size=(3,7,7), padding=(0,3,3))
        self.encoder=nn.Sequential(
            nn.Conv2d(12, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.se1=SEBlock(32)
        self.decoder=nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            # nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x = x.unsqueeze(1)
        x_e_h_n = x[:,:,:3,:,:]
        x_e_h_p = x[:,:,[0,1,3],:,:]
        x2 = self.conv3x3x3(x_e_h_n)
        x3 = self.conv3x5x5(x_e_h_n)
        x4 = self.conv3x7x7(x_e_h_n)
        x5 = self.conv3x3x3(x_e_h_p)
        x6 = self.conv3x5x5(x_e_h_p)
        x7 = self.conv3x7x7(x_e_h_p)
        x = torch.cat((x2,x3,x4,x5,x6,x7),dim=1).view(-1,12,56,56)
        x=self.encoder(x)
        x=self.se1(x)
        x=self.decoder(x)
        return x
