
import math
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
NUM_HEADS = 12
NUM_BLOCKS = 12
MLP_RATIO = 4
NUM_CLASSES = 1000

DEVICE = "cpu"  
IMAGE_PATH = "./Assets/8.jpg"  


preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

class PatchEmbed(nn.Module):
    """Patchify (with unfold), keep a 4D patch view for Snapshot #2, then flatten and linearly project."""
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    def forward(self, x):
       
        B, C, H, W = x.shape
     
        unfolded = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size) 
        patches_flat = unfolded.transpose(1, 2)  
        
        patches_4d = patches_flat.view(B, self.num_patches, C, self.patch_size, self.patch_size)  

        x_embed = self.proj(patches_flat)  
        return patches_4d, patches_flat, x_embed


class EncoderBlock(nn.Module):
    """One ViT encoder block with explicit Q, K, V & attention math exposed for debugging."""
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

      
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        
        self.norm2 = nn.LayerNorm(dim)

        
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

       
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
       
        x_in = x

      
        x_norm1 = self.norm1(x_in)
        qkv = self.qkv(x_norm1)  
        B, N, _ = qkv.shape

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  
       

        attn_probs = attn_scores.softmax(dim=-1)  

        attn_ctx = attn_probs @ v  
        attn_concat = attn_ctx.transpose(1, 2).reshape(B, N, self.dim)  

        attn_out = self.proj(attn_concat)  
        x_after_attn = x_in + attn_out
        x_post_attn_norm = self.norm2(x_after_attn)  

       
        ff_in = x_post_attn_norm
        

        ff_hidden = self.act(self.fc1(ff_in))  

        ff_out = self.fc2(ff_hidden) 

        x_after_mlp = ff_in + ff_out
        x_post_mlp_norm = self.norm3(x_after_mlp)  
        block_out = x_post_mlp_norm
        

        return block_out


class ViT(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE,
                 patch_size=PATCH_SIZE,
                 in_chans=3,
                 embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS,
                 num_blocks=NUM_BLOCKS,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2

    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(num_blocks)])
        self.head = nn.Linear(embed_dim, num_classes)

        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
       
        patches_4d, patches_flat, x_embed = self.patch_embed(x) 

       
        B = x.size(0)
        cls_tok = self.cls_token.expand(B, -1, -1)  
      
        x_tokens = torch.cat([cls_tok, x_embed], dim=1)  

        x_pos = x_tokens + self.pos_embed  


        x_block1_out = self.blocks[0](x_pos)        
        x_block2_out = self.blocks[1](x_block1_out)

        x_curr = x_block2_out
        for i in range(2, len(self.blocks)):
            x_curr = self.blocks[i](x_curr)

        
        x_last = x_curr

      
        final_seq = x_last

      
        cls_rep = final_seq[:, 0, :]  

        logits = self.head(cls_rep)  

        probs = logits.softmax(dim=-1)  

        return {
            "S02_patches4d": patches_4d,
            "S03_patches_flat": patches_flat,
            "S04_patch_embed": x_embed,
            "final_seq": final_seq,
            "cls_rep": cls_rep,
            "logits": logits,
            "probs": probs
        }


def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = preprocess(img).unsqueeze(0)  
    return t

def main():
   
    img_tensor = load_image_tensor(IMAGE_PATH)
    

    model = ViT().to(DEVICE)
    img_tensor = img_tensor.to(DEVICE)

   
    outputs = model(img_tensor)

if __name__ == "__main__":
    main()
