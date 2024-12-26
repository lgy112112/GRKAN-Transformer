import torch
import torch.nn as nn
import torch.nn.functional as F

from kat_rational import KAT_Group  # 假设与之前的KAT_Group在同一项目中

class GroupKANLinear(nn.Module):
    """
    一个示例性“分组 KAN”线性层，将 KAT_Group 与线性映射结合。
    输入/输出形状与 nn.Linear 类似，支持 (batch, length, channels) 三维张量。
    """
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 num_groups: int = 8,
                 bias: bool = True,
                 act_init: str = "gelu"):
        """
        参数:
            in_features  (int): 输入特征通道数 (channels)。
            out_features (int): 输出特征通道数 (channels)。
            num_groups   (int): 分组数量，用于 KAT_Group。
            bias         (bool): 是否在线性层中使用偏置。
            act_init     (str): KAT_Group 的初始化方式，如 "gelu", "relu", "identity" 等。
        """
        super().__init__()
        
        # 1) 定义分组有理函数层
        #    - 负责在 (batch, length) 维度不变的情况下，对每个通道进行分组
        #    - 并对输入进行多项式/有理函数变换
        self.kat_group = KAT_Group(
            num_groups=num_groups,
            mode=act_init,   # 例如 "gelu" 或其他
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 2) 定义线性层，将 in_features -> out_features
        #    - 类似 nn.Linear(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # x.shape: (B, C)
        B, C = x.shape

        # 临时 expand 成 3D => (B, 1, C)
        x = x.unsqueeze(1)

        # 经过 KAT_Group => 还是 (B, 1, C)
        x = self.kat_group(x)

        # 回到 2D => (B, C)
        x = x.squeeze(1)

        # 再做线性层 => (B, out_features)
        x = self.linear(x)
        return x


# ==== 测试用例 ====
if __name__ == "__main__":
    # 例: batch=2, length=10, in_features=16, out_features=32
    test_input = torch.randn(2, 16)  # (B=2, L=10, C=16)
    
    model = GroupKANLinear(in_features=16, out_features=32, num_groups=8, bias=True, act_init="gelu")
    
    # 如果有 GPU 可用，则将模型和数据迁移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_input = test_input.to(device)
    
    # 前向传播
    out = model(test_input)  # (B=2, L=10, out_features=32)
    
    print("Input shape:", test_input.shape)
    print("Output shape:", out.shape)
    
    # import torch
    # import torch.nn as nn

    # # 定义一个简单的线性层
    # linear_layer = nn.Linear(in_features=16, out_features=32)

    # # 创建一个随机输入张量，形状为 (batch_size, in_features)
    # test_input = torch.randn(2, 16)  # 例如 batch_size=2

    # # 前向传播
    # output = linear_layer(test_input)

    # print("Input shape:", test_input.shape)
    # print("Output shape:", output.shape)