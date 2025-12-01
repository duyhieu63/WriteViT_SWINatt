import torch
import torch.nn as nn
from util.util import PosCNN, PositionalEncoding
from params import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding if needed."""
    B, H, W, C = x.shape
    
    # Handle case where H or W is smaller than window_size
    if H < window_size or W < window_size:
        return x.reshape(B, -1, C), H, W
    
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, Hp, Wp

def window_reverse(windows, window_size, H, W, original_h, original_w):
    """Reverse window partition."""
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x[:, :original_h, :original_w, :]
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                 attn_drop=0.0, proj_drop=0.0, spectral=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LocalWindowAttention(Attention):
    """Local window attention with optional cyclic shift"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, 
                 attn_drop=0.0, proj_drop=0.0, spectral=False,
                 window_size=8, shift_size=0):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop, spectral)
        self.window_size = window_size
        self.shift_size = shift_size
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        
        # Fallback to global if spatial dims too small
        if H <= self.window_size and W <= self.window_size:
            return super().forward(x.reshape(B, N, C))
        
        # Pad if needed
        windows, Hp, Wp = window_partition(x, self.window_size)
        
        # Apply attention per window
        attn_windows = super().forward(windows)
        
        # Reverse windows
        x = window_reverse(attn_windows, self.window_size, Hp, Wp, H, W)
        x = x.reshape(B, N, C)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, spectral=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        if spectral:
            self.fc1 = spectral_norm(nn.Linear(in_features, hidden_features))
            self.fc2 = spectral_norm(nn.Linear(hidden_features, out_features))
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, attn_class=Attention, attn_kwargs=None,
                 mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0,
                 init_values=None, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, spectral=False, use_dwconv=True, H=None, W=None):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.H = H
        self.W = W
        
        # Initialize attention with kwargs if provided
        attn_kwargs = attn_kwargs or {}
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop,
                               spectral=spectral, **attn_kwargs)
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, spectral=spectral)
        
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # DWConv branch
        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.bn_dwconv = nn.BatchNorm2d(dim)
            self.act_dwconv = act_layer()

    def forward(self, x):
        B, N, C = x.shape
        
        # Attention path
        attn_input = self.norm1(x)
        if hasattr(self.attn, 'window_size'):
            attn_out = self.attn(attn_input, self.H, self.W)
        else:
            attn_out = self.attn(attn_input)
        
        x_transformer = x + self.drop_path1(self.ls1(attn_out))
        x_transformer = x_transformer + self.drop_path2(self.ls2(self.mlp(self.norm2(x_transformer))))
        
        # DWConv path
        if self.use_dwconv and self.H is not None and self.W is not None:
            x_conv = x_transformer.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
            x_conv = self.dwconv(x_conv)
            x_conv = self.bn_dwconv(x_conv)
            x_conv = self.act_dwconv(x_conv)
            x_conv = x_conv.permute(0, 2, 3, 1).reshape(B, N, C)
            x = x_transformer + x_conv
        else:
            x = x_transformer
            
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
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
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

class Writer(nn.Module):
    def __init__(self, num_classes=NUM_WRITERS, embed_dim=128, num_heads=2,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, max_num_patch=1000,
                 use_dwconv=True, window_size=8):
        super().__init__()
        self.embed_dim = embed_dim
        depth = 3
        self.use_dwconv = use_dwconv
        
        self.layer_norm = LayerNorm()
        patch_size = 4
        self.patch =nn.Conv2d(
            1,
             self.embed_dim,
             kernel_size=patch_size * 2,
             stride=patch_size,
             padding=patch_size // 2,
         )
        self.pos_block = PosCNN(embed_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, drop_rate, max_num_patch)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Progressive drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Build blocks with local attention for first 2, global for last
        self.downsample_blocks = nn.ModuleList()
        for i in range(depth):
            if i < depth - 1:
                # Local window attention for early blocks
                attn_kwargs = {'window_size': window_size, 'shift_size': 0}
                attn_class = LocalWindowAttention
            else:
                # Global attention for final style aggregation
                attn_kwargs = {}
                attn_class = Attention
                
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                attn_class=attn_class,
                attn_kwargs=attn_kwargs,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                spectral=False,
                use_dwconv=use_dwconv,
                H=None,
                W=None
            )
            self.downsample_blocks.append(block)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.initialize_weights()
        
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y=None, training=True):
        # Input: (B, 1, 32, 128) for IAM
        x = self.layer_norm(x)
        x = self.patch(x)  # -> (B, 128, 8, 32)
        
        b, c, h, w = x.shape
        
        # Update H, W for blocks
        if self.use_dwconv:
            for block in self.downsample_blocks:
                if hasattr(block, 'H'):
                    block.H = h
                    block.W = w
        
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, N, C) where N=256
        x = self.pos_enc(x)  # Add positional encoding
        
        for j, blk in enumerate(self.downsample_blocks):
            x = blk(x)
            if j == 0:
                x = self.pos_block(x, h, w)  # PEG after first block
        
        # Head
        x = self.norm(x)  # (B, N, C)
        feature = x
        
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = torch.flatten(x, 1)
        output = self.head(x)
        
        if training:
            loss = self.cross_entropy(output, y.long())
            return feature, loss
        else:
            return feature
class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        results = []
        for item in text:
            item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
            results.append(result)
            result = []

        return (torch.nn.utils.rnn.pad_sequence([torch.LongTensor(text) for text in results], batch_first=True), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
        