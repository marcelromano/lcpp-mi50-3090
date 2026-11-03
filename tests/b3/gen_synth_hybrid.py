#!/usr/bin/env python3
"""Synthetic tiny hybrid (gated-delta-net recurrent + full attention) GGUF, arch qwen35.

Random fixed-seed weights, minimal SPM tokenizer with byte fallback. Uses the gguf-py of the
same source tree (no pip install). Usage: gen_synth_hybrid.py OUT.gguf [SRC_TREE]
Layers: 4 (interval 4 -> layers 0,1,2 recurrent, layer 3 full attention).
"""
import sys
import numpy as np

out = sys.argv[1]
tree = sys.argv[2] if len(sys.argv) > 2 else "/tmp/llama-fork"
sys.path.insert(0, tree + "/gguf-py")
import gguf  # noqa: E402

rng = np.random.default_rng(20261018)

N_EMBD, N_LAYER, N_FF = 64, 4, 128
N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 16
D_STATE, N_GROUP, DT_RANK, D_CONV = 16, 2, 4, 4
D_INNER = D_STATE * DT_RANK                  # 64 (head_v_dim == D_STATE)
KEY_DIM = D_STATE * N_GROUP                  # 32
CONV_DIM = 2 * KEY_DIM + D_INNER             # 128
INTERVAL = 4

# --- vocab: unk, bos, eos, 256 byte tokens, letters, digits, space-prefixed letters
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

w = gguf.GGUFWriter(out, "qwen35")
w.add_name("synthetic-hybrid-b3")
w.add_context_length(512)
w.add_embedding_length(N_EMBD)
w.add_block_count(N_LAYER)
w.add_feed_forward_length(N_FF)
w.add_head_count(N_HEAD)
w.add_head_count_kv(N_HEAD_KV)
w.add_key_length(HEAD_DIM)
w.add_value_length(HEAD_DIM)
w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(8)
w.add_rope_dimension_sections([2, 1, 1, 0])
w.add_rope_freq_base(10000.0)
w.add_ssm_conv_kernel(D_CONV)
w.add_ssm_inner_size(D_INNER)
w.add_ssm_state_size(D_STATE)
w.add_ssm_time_step_rank(DT_RANK)
w.add_ssm_group_count(N_GROUP)
w.add_full_attention_interval(INTERVAL)
w.add_file_type(gguf.LlamaFileType.ALL_F32)

w.add_tokenizer_model("llama")
w.add_tokenizer_pre("default")
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)
w.add_add_bos_token(True)
w.add_add_eos_token(False)
w.add_add_space_prefix(True)

def mat(*shape, scale=0.08):
    # gguf shape {ne0, ne1} == numpy (ne1, ne0)
    return (rng.standard_normal(shape[::-1]) * scale).astype(np.float32)
def put(name, arr):
    w.add_tensor(name, arr)

put("token_embd.weight", mat(N_EMBD, n_vocab, scale=0.2))
put("output_norm.weight", np.ones(N_EMBD, np.float32))
put("output.weight", mat(N_EMBD, n_vocab, scale=0.2))
for il in range(N_LAYER):
    p = "blk.%d." % il
    put(p + "attn_norm.weight", np.ones(N_EMBD, np.float32))
    put(p + "post_attention_norm.weight", np.ones(N_EMBD, np.float32))
    if (il + 1) % INTERVAL == 0:
        put(p + "attn_q.weight", mat(N_EMBD, HEAD_DIM * N_HEAD * 2))
        put(p + "attn_k.weight", mat(N_EMBD, HEAD_DIM * N_HEAD_KV))
        put(p + "attn_v.weight", mat(N_EMBD, HEAD_DIM * N_HEAD_KV))
        put(p + "attn_output.weight", mat(HEAD_DIM * N_HEAD, N_EMBD))
        put(p + "attn_q_norm.weight", np.ones(HEAD_DIM, np.float32))
        put(p + "attn_k_norm.weight", np.ones(HEAD_DIM, np.float32))
    else:
        put(p + "attn_qkv.weight", mat(N_EMBD, CONV_DIM))
        put(p + "attn_gate.weight", mat(N_EMBD, D_INNER))
        put(p + "ssm_conv1d.weight", mat(D_CONV, CONV_DIM, scale=0.3))
        put(p + "ssm_dt.bias", (rng.standard_normal(DT_RANK) * 0.1).astype(np.float32))
        put(p + "ssm_a", (-np.abs(rng.standard_normal(DT_RANK)) - 0.5).astype(np.float32))
        put(p + "ssm_beta.weight", mat(N_EMBD, DT_RANK))
        put(p + "ssm_alpha.weight", mat(N_EMBD, DT_RANK))
        put(p + "ssm_norm.weight", np.ones(D_STATE, np.float32))
        put(p + "ssm_out.weight", mat(D_INNER, N_EMBD))
    put(p + "ffn_gate.weight", mat(N_EMBD, N_FF))
    put(p + "ffn_down.weight", mat(N_FF, N_EMBD))
    put(p + "ffn_up.weight", mat(N_EMBD, N_FF))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "n_vocab", n_vocab)
