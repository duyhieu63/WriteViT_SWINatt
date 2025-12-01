
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Attention import CrossBlock
from util.util import PositionalEncoding, PosCNN
from .blocks import Conv2dBlock, ResBlocks, ActFirstResBlock
from .Unifront import UnifontModule
from params import *
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        spectral=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        if spectral:
            self.qkv = spectral_norm(nn.Linear(dim, dim * 3, bias=qkv_bias))
            self.proj = spectral_norm(nn.Linear(dim, dim))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        spectral=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        if spectral:
            self.fc1 = spectral_norm(nn.Linear(in_features, hidden_features))
            self.fc2 = spectral_norm(nn.Linear(hidden_features, out_features))
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class BlockWithDWConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        spectral=False,
        use_dwconv=True,  # Thêm tùy chọn DWConv
        H=None,            # Chiều cao spatial
        W=None,            # Chiều rộng spatial
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            spectral=spectral,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            spectral=spectral,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Thêm DWConv branch
        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.bn_dwconv = nn.BatchNorm2d(dim)
            self.act_dwconv = act_layer()
            self.H = H
            self.W = W

    def forward(self, x):
        B, N, C = x.shape
        
        # Main transformer path
        x_transformer = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x_transformer = x_transformer + self.drop_path2(self.ls2(self.mlp(self.norm2(x_transformer))))
        
        # DWConv parallel path (nếu được kích hoạt)
        if self.use_dwconv and self.H is not None and self.W is not None:
            # Reshape để áp dụng conv 2D: (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            x_conv = x_transformer.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
            
            # Áp dụng depthwise convolution
            x_conv = self.dwconv(x_conv)
            x_conv = self.bn_dwconv(x_conv)
            x_conv = self.act_dwconv(x_conv)
            
            # Reshape trở lại: (B, C, H, W) -> (B, H, W, C) -> (B, N, C)
            x_conv = x_conv.permute(0, 2, 3, 1).reshape(B, N, C)
            
            # Fusion: kết hợp transformer path và conv path
            x = x_transformer + x_conv
        else:
            x = x_transformer
            print("Not using DWConv path generator")
            
        return x
def DropPath(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return DropPath(x, self.drop_prob, self.training)

def spectral_norm(module):
    return nn.utils.spectral_norm(module)

#DWConv Generator
class Generator(nn.Module):
    def __init__(
        self,
        arg=None,
        embed_dim=128,
        depth=3,
        num_heads=2,
        mlp_ratio=4,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        max_num_patch=100,
        use_dwconv=True,  # Thêm tùy chọn DWConv
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = None
        self.grid_size = None
        self.embed_dim = [128, 128, 64, 64, 32, 32, 16]
        num_block = 4
        self.use_dwconv = use_dwconv
        
        self.pos_enc = PositionalEncoding(embed_dim, drop, max_num_patch)
        self.query_embed = UnifontModule(
            embed_dim,
            ALPHABET,
            input_type="unifont",
            linear=True,
        )

        """Block 1 - Giữ nguyên CrossBlock"""
        index = 1
        self.blocks_2 = nn.ModuleList(
            [
                CrossBlock(
                    self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth+2)
            ]
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_1 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_1 = self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])

        """Block 2 - Thay bằng BlockWithDWConv"""
        index += 1
        self.blocks_3 = nn.ModuleList(
            [
                BlockWithDWConv(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop,
                    attn_drop=drop,
                    norm_layer=norm_layer,
                    use_dwconv=use_dwconv,  # Bật DWConv
                    H=None,  # Sẽ được cập nhật trong forward
                    W=None,
                )
                for i in range(depth)
            ]
        )
        self.layer_norm3 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_2 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_2 = self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])

        """Block 3 - Thay bằng BlockWithDWConv"""
        index += 1
        self.blocks_4 = nn.ModuleList(
            [
                BlockWithDWConv(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop,
                    attn_drop=drop,
                    norm_layer=norm_layer,
                    use_dwconv=use_dwconv,  # Bật DWConv
                    H=None,
                    W=None,
                )
                for i in range(depth)
            ]
        )
        self.layer_norm4 = nn.LayerNorm(self.embed_dim[index])
        self.tRGB_3 = nn.Sequential(
            nn.Conv2d(self.embed_dim[index], self.embed_dim[num_block], 3, 1, 1)
        )
        self.conv_3 = self._make_upsample_block(self.embed_dim[index], self.embed_dim[index+1])

        """Block 4 - Thay bằng BlockWithDWConv"""
        index += 1
        self.blocks_5 = nn.ModuleList(
            [
                BlockWithDWConv(
                    dim=self.embed_dim[index],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop,
                    attn_drop=drop,
                    norm_layer=norm_layer,
                    use_dwconv=use_dwconv,  # Bật DWConv
                    H=None,
                    W=None,
                )
                for i in range(depth)
            ]
        )
        self.layer_norm5 = nn.LayerNorm(self.embed_dim[index])

        self.pos_block = nn.ModuleList([PosCNN(i, i) for i in self.embed_dim])
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.noise = torch.distributions.Normal(
            loc=torch.tensor([0.0]), scale=torch.tensor([1.0])
        )

        self.deconv = nn.Sequential(
            ResBlocks(
                2, self.embed_dim[index], norm="in", activation="relu", pad_type="reflect"
            ),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(
                self.embed_dim[index],
                self.embed_dim[index + 1],
                3,
                1,
                1,
                norm="in",
                activation="none",
                pad_type="reflect",
            ),
            Conv2dBlock(
                self.embed_dim[5],
                self.embed_dim[5],
                5,
                1,
                2,
                norm="in",
                activation="relu",
                pad_type="reflect",
            ),
            Conv2dBlock(
                self.embed_dim[5],
                1,
                7,
                1,
                3,
                norm="none",
                activation="tanh",
                pad_type="reflect",
            ),
        )
        self.initialize_weights()

    def _make_upsample_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim, out_dim, 3, 1, 1, norm="in", activation="none", pad_type="reflect"),
            Conv2dBlock(out_dim, out_dim, 3, 1, 1, norm="in", activation="relu", pad_type="reflect"),
        )

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):  # Thêm khởi tạo cho BatchNorm2d của DWConv
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _generate_features(self, src, tgt):
        b = src.size(0)
        start_h = 2
        start_w = tgt.size(1)

        src = src
        tmp = self.query_embed(tgt.clone())
        tgt = self.pos_enc(self.query_embed(tgt))

        stack_output = []
        for blk in self.blocks_2:
            tgt = blk(tgt, src)
            stack_output.append(tgt)
        h2 = stack_output[-1]

        tgt = torch.cat([h2, tmp], dim=1)
        tgt = self.layer_norm2(tgt)
        tgt = tgt.permute(0, 2, 1).view(b, self.embed_dim[1], start_h, start_w)
        x_1 = self.tRGB_1(tgt)

        tgt = self.conv_1(tgt)
        b, c, h, w = tgt.shape
        
        # Cập nhật kích thước spatial cho blocks_3
        if self.use_dwconv:
            for blk in self.blocks_3:
                blk.H = h
                blk.W = w
                
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_3):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[2](tgt, h, w)
        tgt = self.layer_norm3(tgt).permute(0, 2, 1).view(b, self.embed_dim[2], h, w)
        x_2 = self.tRGB_2(tgt)

        tgt = self.conv_2(tgt)
        b, c, h, w = tgt.shape
        
        # Cập nhật kích thước spatial cho blocks_4
        if self.use_dwconv:
            for blk in self.blocks_4:
                blk.H = h
                blk.W = w
                
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_4):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[3](tgt, h, w)
        tgt = self.layer_norm4(tgt).permute(0, 2, 1).view(b, self.embed_dim[3], h, w)
        x_3 = self.tRGB_3(tgt)

        tgt = self.conv_3(tgt)
        b, c, h, w = tgt.shape
        
        # Cập nhật kích thước spatial cho blocks_5
        if self.use_dwconv:
            for blk in self.blocks_5:
                blk.H = h
                blk.W = w
                
        tgt = tgt.view(b, c, -1).permute(0, 2, 1)

        for j, blk in enumerate(self.blocks_5):
            tgt = blk(tgt)
            if j == 0:
                tgt = self.pos_block[4](tgt, h, w)
        tgt = self.layer_norm5(tgt).permute(0, 2, 1).view(b, self.embed_dim[4], h, w)

        fused = (
            F.interpolate(x_1, scale_factor=8)
            + F.interpolate(x_2, scale_factor=4)
            + F.interpolate(x_3, scale_factor=2)
            + tgt
        )
        noise = self.noise.sample(fused.size()).squeeze(-1).to(fused.device)
        return fused + noise

    def forward(self, src_w, tgt):
        features = self._generate_features(src_w, tgt)
        return self.deconv(features)

    def Eval(self, xw, QRS):
        outputs = []
        for i in range(QRS.shape[1]):
            tgt = QRS[:, i, :].squeeze(1)
            features = self._generate_features(xw, tgt)
            outputs.append(self.deconv(features).detach())
        return outputs