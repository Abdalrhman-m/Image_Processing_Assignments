from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

torch.manual_seed(7)

SRC_SENTENCE = "hello from cairo egypt team project"
TGT_SENTENCE = "we are debugging transformer today"

def build_vocab_from_sentences(*sentences):
    words = []
    for s in sentences:
        words.extend(s.strip().split())
    vocab = {w:i for i, w in enumerate(sorted(set(words)))}
    return vocab

src_vocab = build_vocab_from_sentences(SRC_SENTENCE)
tgt_vocab = build_vocab_from_sentences(TGT_SENTENCE)

def encode(sentence, vocab):
    return [vocab[w] for w in sentence.strip().split()]

def decode(ids, inv_vocab):
    return " ".join(inv_vocab[i] for i in ids)

src_ids = encode(SRC_SENTENCE, src_vocab)    
tgt_ids = encode(TGT_SENTENCE, tgt_vocab)     

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)


inv_src_vocab = {i:w for w,i in src_vocab.items()}
inv_tgt_vocab = {i:w for w,i in tgt_vocab.items()}

@dataclass
class HParams:
    d_model: int = 128
    nhead: int = 4
    d_ff: int = 256
    num_enc_layers: int = 2
    num_dec_layers: int = 2

hp = HParams()
assert hp.d_model % hp.nhead == 0
d_head = hp.d_model // hp.nhead

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
    
        b, s, _ = x.size()
        x = x.view(b, s, self.nhead, self.d_head).transpose(1, 2)  # (batch, heads, seq, d_head)
        return x

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, mask: torch.Tensor | None = None):
   
        q = self.W_q(q_in)   
        k = self.W_k(k_in)   
        v = self.W_v(v_in)  
      
        q_heads = self._split_heads(q)  
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        scores_pre_mask = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(self.d_head) 
      
        scores = scores_pre_mask.clone()
        if mask is not None:
          
            scores = scores + mask 

        attn_weights = F.softmax(scores, dim=-1)  
        attn_out_heads = attn_weights @ v_heads  
        attn_concat = attn_out_heads.transpose(1, 2).contiguous().view(q_in.size(0), q_in.size(1), self.d_model)
       
        out = self.W_o(attn_concat)
        return out, q, k, v, q_heads, k_heads, v_heads, scores_pre_mask, attn_weights, attn_concat


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        ff_in = x                      
        x1 = self.fc1(ff_in)           
        x2 = F.relu(x1)
        x3 = self.fc2(x2)             
        return x3, ff_in, x1, x3



class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, capture: bool = False):
       
        enc_in = x                         

        attn_out, q, k, v, qh, kh, vh, scores_pre, attn_w, attn_concat = self.mha(x, x, x, mask=None)
        
        res1_a = enc_in                  
        res1_b = attn_out                  
        x = self.ln1(res1_a + res1_b)    

        ffn_out, ff_in, ff1, ff2 = self.ffn(x)
       
        res2_a = x
        res2_b = ffn_out
        enc_out = self.ln2(res2_a + res2_b)  

        if capture:
            
            trace = dict(
                enc_in=enc_in,
                q=q, k=k, v=v,
                q_heads=qh, k_heads=kh, v_heads=vh,
                scores_pre_mask=scores_pre,
                attn_weights=attn_w,
                attn_concat=attn_concat,
                res1_a=res1_a, res1_b=res1_b,
                ln1_out=x,        
                ff_in=ff_in, ff1=ff1, ff2=ff2,
                enc_out=enc_out
            )
            return enc_out, trace
        return enc_out, None


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, nhead)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, subsequent_mask, capture: bool = False):
      
        msa_out, q_m, k_m, v_m, qh_m, kh_m, vh_m, scores_pre_m, attn_w_m, attn_concat_m = \
            self.self_mha(x, x, x, mask=subsequent_mask)
        res1_a = x
        res1_b = msa_out
        x = self.ln1(res1_a + res1_b)  

       
        ca_out, q_c, k_c, v_c, qh_c, kh_c, vh_c, scores_pre_c, attn_w_c, attn_concat_c = \
            self.cross_mha(x, enc_out, enc_out, mask=None)
        res2_a = x
        res2_b = ca_out
        x = self.ln2(res2_a + res2_b)  

      
        ffn_out, ff_in, ff1, ff2 = self.ffn(x)
        res3_a = x
        res3_b = ffn_out
        dec_out = self.ln3(res3_a + res3_b) 

        if capture:
            trace = dict(
              
                q_m=q_m, k_m=k_m, v_m=v_m,
                q_heads_m=qh_m, k_heads_m=kh_m, v_heads_m=vh_m,
                scores_pre_mask_m=scores_pre_m,
                attn_weights_m=attn_w_m,
                attn_concat_m=attn_concat_m,
                after_msa_norm=x, 

               
                q_c=q_c, k_c=k_c, v_c=v_c,
                q_heads_c=qh_c, k_heads_c=kh_c, v_heads_c=vh_c,
                scores_pre_mask_c=scores_pre_c,
                attn_weights_c=attn_w_c,
                attn_concat_c=attn_concat_c,
                after_ca_norm=x,

              
                ff_in=ff_in, ff1=ff1, ff2=ff2,
                dec_out=dec_out
            )
            return dec_out, trace
        return dec_out, None


class TinyTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, hp: HParams):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, hp.d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, hp.d_model)
        self.pos = PositionalEncoding(hp.d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(hp.d_model, hp.nhead, hp.d_ff) for _ in range(hp.num_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(hp.d_model, hp.nhead, hp.d_ff) for _ in range(hp.num_dec_layers)])
        self.proj = nn.Linear(hp.d_model, tgt_vocab_size)

    def make_subsequent_mask(self, tgt_len: int, batch_size: int = 1) -> torch.Tensor:
        
        mask = torch.full((tgt_len, tgt_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
       
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return mask

    def forward(self, src_ids, tgt_ids):
   
        device = next(self.parameters()).device
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0) 
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0) 

      
        emb_weight_slice = self.src_emb.weight[:5, :5] 

        
        src_embed = self.src_emb(src)                  
        src_with_pos = self.pos(src_embed)            

        tgt_embed = self.tgt_emb(tgt)
        tgt_with_pos = self.pos(tgt_embed)

     
        x = src_with_pos
        enc_traces = []
        for li, enc in enumerate(self.enc_layers):
            cap = (li == 0)  
            x, enc_trace = enc(x, capture=cap)
            enc_traces.append(enc_trace)
        enc_out = x 

       
        y = tgt_with_pos
        tgt_mask = self.make_subsequent_mask(tgt.size(1), batch_size=tgt.size(0)) 
        dec_traces = []
        for li, dec in enumerate(self.dec_layers):
            cap = (li == 0)  
            y, dec_trace = dec(y, enc_out, tgt_mask, capture=cap)
            dec_traces.append(dec_trace)

        dec_final = y                                
        logits = self.proj(dec_final)                  
        logits_slice = logits[0, 0, :10]               

       
        snapshot_refs = dict(
          
            emb_weight_slice=emb_weight_slice,  
            src_embed=src_embed,               
            src_with_pos=src_with_pos,          
            enc_traces=enc_traces,             
            tgt_with_pos=tgt_with_pos,          
            tgt_mask=tgt_mask,                  
            dec_traces=dec_traces,             
            dec_final=dec_final,                
            logits=logits,                      
            logits_slice=logits_slice           
        )
        return logits, snapshot_refs


def main():
    model = TinyTransformer(src_vocab_size, tgt_vocab_size, hp)
    logits, refs = model(src_ids, tgt_ids)
  
    stop_here_for_debugger = True 

if __name__ == "__main__":
    main()
