import torch
import torch.nn.functional as F
from torch import nn

# 从现有模型文件中导入一些基础组件
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

class CrossAttention(nn.Module):
    """简单的跨模态注意力机制"""
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        # x1作为query，x2作为key和value
        batch_size, C, H, W = x1.size()
        
        # 生成查询、键、值
        queries = self.query(x1).view(batch_size, C, -1)  # (B, C, H*W)
        keys = self.key(x2).view(batch_size, C, -1)       # (B, C, H*W)
        values = self.value(x2).view(batch_size, C, -1)   # (B, C, H*W)
        
        # 计算注意力
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys)  # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)
        
        out = torch.bmm(values, attention_scores.permute(0, 2, 1))  # (B, C, H*W)
        return out.view(batch_size, C, H, W)

class ElementwiseFusion(nn.Module):
    """元素级融合模块"""
    def __init__(self, channels):
        super(ElementwiseFusion, self).__init__()
        # 修复：sum(32) + diff(32) + prod(32) + concat(64) = 160
        self.conv = nn.Conv2d(channels * 5, channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # 计算元素级操作
        sum_feat = x1 + x2
        diff_feat = x1 - x2
        prod_feat = x1 * x2
        concat_feat = torch.cat([x1, x2], dim=1)
        
        # 拼接所有特征
        combined = torch.cat([sum_feat, diff_feat, prod_feat, concat_feat], dim=1)
        return self.conv(combined)

# ============================================================================
# 模型1: 简单的双分支网络 (Multi-Branch Basic)
# ============================================================================
class MB_Basic(nn.Module):
    """基础多分支网络：分别处理tracker和calorimeter数据"""
    def __init__(self):
        super(MB_Basic, self).__init__()
        
        # Tracker分支 (处理trkn, trkp)
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支 (处理emcal, hcal)
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        batch_size = x.size(0)
        
        # 分离数据
        calorimeter_data = x[:, :2, :, :]  # emcal, hcal
        tracker_data = x[:, 2:, :, :]      # trkn, trkp
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 简单拼接融合
        fused_features = torch.cat([calorimeter_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型2: 带注意力融合的多分支网络 (Multi-Branch with Attention)
# ============================================================================
class MB_Attention(nn.Module):
    """带注意力机制的多分支网络"""
    def __init__(self):
        super(MB_Attention, self).__init__()
        
        # Tracker分支
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 跨模态注意力
        self.cross_attention = CrossAttention(32)
        
        # SE注意力
        self.se_block = SEBlock(32)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        calorimeter_data = x[:, :2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 跨模态注意力：让tracker特征关注calorimeter特征
        attended_features = self.cross_attention(tracker_features, calorimeter_features)
        
        # SE注意力
        attended_features = self.se_block(attended_features)
        
        # 融合
        fused = self.fusion(attended_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型3: 元素级融合的多分支网络 (Multi-Branch with Elementwise Fusion)
# ============================================================================
class MB_Elementwise(nn.Module):
    """使用元素级操作融合的多分支网络"""
    def __init__(self):
        super(MB_Elementwise, self).__init__()
        
        # Tracker分支
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 元素级融合
        self.elementwise_fusion = ElementwiseFusion(32)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        calorimeter_data = x[:, :2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 元素级融合
        fused_features = self.elementwise_fusion(tracker_features, calorimeter_features)
        
        # 解码输出
        output = self.decoder(fused_features)
        
        return output

# ============================================================================
# 模型4: 三分支网络 (Three-Branch Network)
# ============================================================================
class MB_ThreeBranch(nn.Module):
    """三分支网络：分别处理EMCal、HCal和Tracker"""
    def __init__(self):
        super(MB_ThreeBranch, self).__init__()
        
        # EMCal分支
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # HCal分支
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Tracker分支
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        
        # 分离数据
        emcal_data = x[:, 0:1, :, :]  # (B, 1, 56, 56)
        hcal_data = x[:, 1:2, :, :]   # (B, 1, 56, 56)
        tracker_data = x[:, 2:, :, :]  # (B, 2, 56, 56)
        
        # 分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 拼接融合
        fused_features = torch.cat([emcal_features, hcal_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型5: 渐进式融合网络 (Progressive Fusion)
# ============================================================================
class MB_Progressive(nn.Module):
    """渐进式融合网络：先融合相似模态，再融合不同模态"""
    def __init__(self):
        super(MB_Progressive, self).__init__()
        
        # 第一阶段：分别处理
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        
        # 第二阶段：融合calorimeter数据
        self.calorimeter_fusion = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 16+16=32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 第三阶段：最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),  # 32+16=48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        emcal_data = x[:, 0:1, :, :]
        hcal_data = x[:, 1:2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 第一阶段：分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 第二阶段：融合calorimeter数据
        calorimeter_features = torch.cat([emcal_features, hcal_features], dim=1)
        calorimeter_fused = self.calorimeter_fusion(calorimeter_features)
        
        # 第三阶段：最终融合
        final_features = torch.cat([calorimeter_fused, tracker_features], dim=1)
        final_fused = self.final_fusion(final_features)
        
        # 解码输出
        output = self.decoder(final_fused)
        
        return output

# ============================================================================
# V2版本模型：扩大一个级别的参数量
# ============================================================================

# ============================================================================
# 模型1 V2: 基础多分支网络 (Multi-Branch Basic V2)
# ============================================================================
class MB_Basic_v2(nn.Module):
    """基础多分支网络V2：扩大参数量"""
    def __init__(self):
        super(MB_Basic_v2, self).__init__()
        
        # Tracker分支 (处理trkn, trkp) - 扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支 (处理emcal, hcal) - 扩大参数量
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 融合层 - 扩大参数量
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64+64=128, 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        batch_size = x.size(0)
        
        # 分离数据
        calorimeter_data = x[:, :2, :, :]  # emcal, hcal
        tracker_data = x[:, 2:, :, :]      # trkn, trkp
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 简单拼接融合
        fused_features = torch.cat([calorimeter_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型3 V2: 元素级融合的多分支网络 (Multi-Branch with Elementwise Fusion V2)
# ============================================================================
class MB_Elementwise_v2(nn.Module):
    """使用元素级操作融合的多分支网络V2：扩大参数量"""
    def __init__(self):
        super(MB_Elementwise_v2, self).__init__()
        
        # Tracker分支 - 扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支 - 扩大参数量
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 元素级融合 - 扩大参数量
        self.elementwise_fusion = ElementwiseFusion_v2(64)  # 32->64
        
        # 解码器 - 扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        calorimeter_data = x[:, :2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 元素级融合
        fused_features = self.elementwise_fusion(tracker_features, calorimeter_features)
        
        # 解码输出
        output = self.decoder(fused_features)
        
        return output

# 扩大的元素级融合模块
class ElementwiseFusion_v2(nn.Module):
    """元素级融合模块V2：扩大参数量"""
    def __init__(self, channels):
        super(ElementwiseFusion_v2, self).__init__()
        # 64*5=320 channels
        self.conv = nn.Conv2d(channels * 5, channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # 计算元素级操作
        sum_feat = x1 + x2
        diff_feat = x1 - x2
        prod_feat = x1 * x2
        concat_feat = torch.cat([x1, x2], dim=1)
        
        # 拼接所有特征
        combined = torch.cat([sum_feat, diff_feat, prod_feat, concat_feat], dim=1)
        return self.conv(combined)

# ============================================================================
# 模型4 V2: 三分支网络 (Three-Branch Network V2)
# ============================================================================
class MB_ThreeBranch_v2(nn.Module):
    """三分支网络V2：扩大参数量"""
    def __init__(self):
        super(MB_ThreeBranch_v2, self).__init__()
        
        # EMCal分支 - 扩大参数量
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # HCal分支 - 扩大参数量
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Tracker分支 - 扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 融合层 - 扩大参数量
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 64+64+64=192, 96->192
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        
        # 分离数据
        emcal_data = x[:, 0:1, :, :]  # (B, 1, 56, 56)
        hcal_data = x[:, 1:2, :, :]   # (B, 1, 56, 56)
        tracker_data = x[:, 2:, :, :]  # (B, 2, 56, 56)
        
        # 分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 拼接融合
        fused_features = torch.cat([emcal_features, hcal_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型5 V2: 渐进式融合网络 (Progressive Fusion V2)
# ============================================================================
class MB_Progressive_v2(nn.Module):
    """渐进式融合网络V2：扩大参数量"""
    def __init__(self):
        super(MB_Progressive_v2, self).__init__()
        
        # 第一阶段：分别处理 - 扩大参数量
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 16->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 第二阶段：融合calorimeter数据 - 扩大参数量
        self.calorimeter_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64+64=128, 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 第三阶段：最终融合 - 扩大参数量
        self.final_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64+64=128, 48->128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        emcal_data = x[:, 0:1, :, :]
        hcal_data = x[:, 1:2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 第一阶段：分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 第二阶段：融合calorimeter数据
        calorimeter_features = torch.cat([emcal_features, hcal_features], dim=1)
        calorimeter_fused = self.calorimeter_fusion(calorimeter_features)
        
        # 第三阶段：最终融合
        final_features = torch.cat([calorimeter_fused, tracker_features], dim=1)
        final_fused = self.final_fusion(final_features)
        
        # 解码输出
        output = self.decoder(final_fused)
        
        return output

# ============================================================================
# V3版本模型：进一步扩大参数量
# ============================================================================

# ============================================================================
# 模型1 V3: 基础多分支网络 (Multi-Branch Basic V3)
# ============================================================================
class MB_Basic_v3(nn.Module):
    """基础多分支网络V3：进一步扩大参数量"""
    def __init__(self):
        super(MB_Basic_v3, self).__init__()
        
        # Tracker分支 (处理trkn, trkp) - 进一步扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支 (处理emcal, hcal) - 进一步扩大参数量
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 融合层 - 进一步扩大参数量
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128+128=256, 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 进一步扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        batch_size = x.size(0)
        
        # 分离数据
        calorimeter_data = x[:, :2, :, :]  # emcal, hcal
        tracker_data = x[:, 2:, :, :]      # trkn, trkp
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 简单拼接融合
        fused_features = torch.cat([calorimeter_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型3 V3: 元素级融合的多分支网络 (Multi-Branch with Elementwise Fusion V3)
# ============================================================================
class MB_Elementwise_v3(nn.Module):
    """使用元素级操作融合的多分支网络V3：进一步扩大参数量"""
    def __init__(self):
        super(MB_Elementwise_v3, self).__init__()
        
        # Tracker分支 - 进一步扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Calorimeter分支 - 进一步扩大参数量
        self.calorimeter_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 元素级融合 - 进一步扩大参数量
        self.elementwise_fusion = ElementwiseFusion_v3(128)  # 64->128
        
        # 解码器 - 进一步扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        calorimeter_data = x[:, :2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 分别编码
        calorimeter_features = self.calorimeter_encoder(calorimeter_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 元素级融合
        fused_features = self.elementwise_fusion(tracker_features, calorimeter_features)
        
        # 解码输出
        output = self.decoder(fused_features)
        
        return output

# 进一步扩大的元素级融合模块
class ElementwiseFusion_v3(nn.Module):
    """元素级融合模块V3：进一步扩大参数量"""
    def __init__(self, channels):
        super(ElementwiseFusion_v3, self).__init__()
        # 128*5=640 channels
        self.conv = nn.Conv2d(channels * 5, channels, kernel_size=1)
        
    def forward(self, x1, x2):
        # 计算元素级操作
        sum_feat = x1 + x2
        diff_feat = x1 - x2
        prod_feat = x1 * x2
        concat_feat = torch.cat([x1, x2], dim=1)
        
        # 拼接所有特征
        combined = torch.cat([sum_feat, diff_feat, prod_feat, concat_feat], dim=1)
        return self.conv(combined)

# ============================================================================
# 模型4 V3: 三分支网络 (Three-Branch Network V3)
# ============================================================================
class MB_ThreeBranch_v3(nn.Module):
    """三分支网络V3：进一步扩大参数量"""
    def __init__(self):
        super(MB_ThreeBranch_v3, self).__init__()
        
        # EMCal分支 - 进一步扩大参数量
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # HCal分支 - 进一步扩大参数量
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Tracker分支 - 进一步扩大参数量
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 融合层 - 进一步扩大参数量
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 128+128+128=384, 192->384
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 进一步扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # x shape: (B, 4, 56, 56) - [emcal, hcal, trkn, trkp]
        
        # 分离数据
        emcal_data = x[:, 0:1, :, :]  # (B, 1, 56, 56)
        hcal_data = x[:, 1:2, :, :]   # (B, 1, 56, 56)
        tracker_data = x[:, 2:, :, :]  # (B, 2, 56, 56)
        
        # 分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 拼接融合
        fused_features = torch.cat([emcal_features, hcal_features, tracker_features], dim=1)
        
        # 融合处理
        fused = self.fusion(fused_features)
        
        # 解码输出
        output = self.decoder(fused)
        
        return output

# ============================================================================
# 模型5 V3: 渐进式融合网络 (Progressive Fusion V3)
# ============================================================================
class MB_Progressive_v3(nn.Module):
    """渐进式融合网络V3：进一步扩大参数量"""
    def __init__(self):
        super(MB_Progressive_v3, self).__init__()
        
        # 第一阶段：分别处理 - 进一步扩大参数量
        self.emcal_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.hcal_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.tracker_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 第二阶段：融合calorimeter数据 - 进一步扩大参数量
        self.calorimeter_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128+128=256, 64->128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 第三阶段：最终融合 - 进一步扩大参数量
        self.final_fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128+128=256, 128->256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 保持128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 新增一层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 解码器 - 进一步扩大参数量
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 新增一层
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        # 分离数据
        emcal_data = x[:, 0:1, :, :]
        hcal_data = x[:, 1:2, :, :]
        tracker_data = x[:, 2:, :, :]
        
        # 第一阶段：分别编码
        emcal_features = self.emcal_encoder(emcal_data)
        hcal_features = self.hcal_encoder(hcal_data)
        tracker_features = self.tracker_encoder(tracker_data)
        
        # 第二阶段：融合calorimeter数据
        calorimeter_features = torch.cat([emcal_features, hcal_features], dim=1)
        calorimeter_fused = self.calorimeter_fusion(calorimeter_features)
        
        # 第三阶段：最终融合
        final_features = torch.cat([calorimeter_fused, tracker_features], dim=1)
        final_fused = self.final_fusion(final_features)
        
        # 解码输出
        output = self.decoder(final_fused)
        
        return output