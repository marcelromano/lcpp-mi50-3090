#!/usr/bin/env python3
"""Synthetic tiny attention-only GGUF (arch llama, 2 layers), random fixed-seed weights.
Stand-in for the absent /tmp/llama-test-attention.gguf in the attention-only kv-admission suites
(two-contexts, bounded-reader, r4-r5). Usage: gen_synth_attn.py OUT.gguf [SRC_TREE]
"""
import sys
from pathlib import Path
import numpy as np

out = sys.argv[1]
tree = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2])
sys.path.insert(0, tree + "/gguf-py")
import gguf  # noqa: E402

rng = np.random.default_rng(20261019)
N_EMBD, N_LAYER, N_FF, N_HEAD, N_HEAD_KV = 64, 2, 128, 4, 2
HD = N_EMBD // N_HEAD

tokens, scores, types = [], [], []
def add(t, s, ty):
    tokens.append(t.encode("utf-8")); scores.append(float(s)); types.append(int(ty))
TT = gguf.TokenType
add("<unk>", 0, TT.UNKNOWN); add("<s>", 0, TT.CONTROL); add("</s>", 0, TT.CONTROL)
for b in range(256):
    add("<0x%02X>" % b, 0, TT.BYTE)
add("▁", -1, TT.NORMAL)
for c in "abcdefghijklmnopqrstuvwxyz0123456789.,":
    add(c, -2, TT.NORMAL)
for c in "abcdefghijklmnopqrstuvwxyz":
    add("▁" + c, -3, TT.NORMAL)
n_vocab = len(tokens)

w = gguf.GGUFWriter(out, "llama")
w.add_name("synthetic-attn-b3")
w.add_context_length(512)
w.add_embedding_length(N_EMBD)
w.add_block_count(N_LAYER)
w.add_feed_forward_length(N_FF)
w.add_head_count(N_HEAD)
w.add_head_count_kv(N_HEAD_KV)
w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(HD)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_tokenizer_model("llama")
w.add_tokenizer_pre("default")
w.add_token_list(tokens); w.add_token_scores(scores); w.add_token_types(types)
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
w.add_add_bos_token(True); w.add_add_eos_token(False)

def mat(*shape, scale=0.08):
    return (rng.standard_normal(shape[::-1]) * scale).astype(np.float32)
def put(name, arr):
    w.add_tensor(name, arr)

put("token_embd.weight", mat(N_EMBD, n_vocab, scale=0.2))
put("output_norm.weight", np.ones(N_EMBD, np.float32))
put("output.weight", mat(N_EMBD, n_vocab, scale=0.2))
for il in range(N_LAYER):
    p = "blk.%d." % il
    put(p + "attn_norm.weight", np.ones(N_EMBD, np.float32))
    put(p + "attn_q.weight", mat(N_EMBD, HD * N_HEAD))
    put(p + "attn_k.weight", mat(N_EMBD, HD * N_HEAD_KV))
    put(p + "attn_v.weight", mat(N_EMBD, HD * N_HEAD_KV))
    put(p + "attn_output.weight", mat(HD * N_HEAD, N_EMBD))
    put(p + "ffn_norm.weight", np.ones(N_EMBD, np.float32))
    put(p + "ffn_gate.weight", mat(N_EMBD, N_FF))
    put(p + "ffn_down.weight", mat(N_FF, N_EMBD))
    put(p + "ffn_up.weight", mat(N_EMBD, N_FF))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "n_vocab", n_vocab)
