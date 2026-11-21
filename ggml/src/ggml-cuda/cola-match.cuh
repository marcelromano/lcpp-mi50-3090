#ifdef GGML_EXP_COLA
#define COLA_FAIL() return 0
static bool ggml_cuda_sk_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 + ggml_nbytes(a) <= b0 || b0 + ggml_nbytes(b) <= a0;
}

static int ggml_cuda_cola_try_gates(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * mm_a = cgraph->nodes[i];
    if (!(ggml_cuda_cola_mask() & GGML_COLA_GATES) || mm_a->op != GGML_OP_MUL_MAT || mm_a->src[0]->type != GGML_TYPE_BF16) {
        COLA_FAIL();
    }
    // full pattern with the sigmoid, or (GGML_COLA bit 32 clear) the pattern up to the beta projection with the sigmoid left alone
    const bool with_sig = (ggml_cuda_cola_mask() & GGML_COLA_SIGK) != 0;
    static const ggml_op want[6] = { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_UNARY };
    const int n_want = with_sig ? 6 : 5;
    int seq[6];
    int n = 0;
    int last = i;
    seq[n++] = i;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 14 && n < n_want; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (t->op != want[n]) {
            COLA_FAIL();
        }
        seq[n++] = j;
        last = j;
    }
    if (n < n_want) {
        COLA_FAIL();
    }
    const ggml_tensor * add  = cgraph->nodes[seq[1]];
    const ggml_tensor * sp   = cgraph->nodes[seq[2]];
    const ggml_tensor * mul  = cgraph->nodes[seq[3]];
    const ggml_tensor * mm_b = cgraph->nodes[seq[4]];
    const ggml_tensor * sig  = with_sig ? cgraph->nodes[seq[5]] : nullptr;
    if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS || (sig && ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID)) {
        COLA_FAIL();
    }
    // data flow
    const ggml_tensor * add_a = add->src[0];
    const ggml_tensor * dt    = add->src[1];
    if (!(add_a == mm_a || add_a->view_src == mm_a) || add_a->data != mm_a->data) {
        COLA_FAIL();
    }
    if (sp->src[0] != add || !(mul->src[0] == sp || mul->src[1] == sp)) {
        COLA_FAIL();
    }
    const ggml_tensor * a = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    if (sig) {
        const ggml_tensor * sig_in = sig->src[0];
        if (!(sig_in == mm_b || sig_in->view_src == mm_b) || sig_in->data != mm_b->data) {
            COLA_FAIL();
        }
    }
    const int64_t nh = mm_a->ne[0];
    const int64_t nt = mm_a->ne[1];
    if (mm_b->src[1] != mm_a->src[1] || mm_b->ne[0] != nh || mm_b->ne[1] != nt || mm_a->ne[2] != 1 || mm_a->ne[3] != 1 ||
        mm_b->ne[2] != 1 || mm_b->ne[3] != 1 || mm_a->src[0]->ne[0] != mm_b->src[0]->ne[0] ||
        mm_a->src[0]->nb[1] != mm_b->src[0]->nb[1] || mm_a->src[0]->type != mm_b->src[0]->type || nh*nt > 960) {
        COLA_FAIL();
    }
    // qwen35 reshapes alpha [nh,nt] to [nh,nt/2,2] in c2.
    // The kernel indexes flattened contiguous columns; dt/a still broadcast per head.
    // Do not relax the c1 path or accept arbitrary strided/batched broadcasts.
    static const bool slots2 = [] {
        const char * e = getenv("GGML_COLA_GATES_SLOTS2");
        return e && std::atoi(e) == 1;
    }();
    const bool flat_slots2 = slots2 && nt <= 8 && nt % 2 == 0 &&
        add->ne[0] == nh && add->ne[1] == nt/2 && add->ne[2] == 2 && add->ne[3] == 1 &&
        ggml_are_same_shape(add_a, add) && ggml_are_same_shape(sp, add) && ggml_are_same_shape(mul, add) &&
        ggml_is_contiguous(add_a) && ggml_is_contiguous(add) && ggml_is_contiguous(sp);
    if (mm_a->type != GGML_TYPE_F32 || mm_b->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || (sig && sig->type != GGML_TYPE_F32) ||
        dt->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || ggml_nelements(dt) != nh || ggml_nelements(a) != nh ||
        !ggml_is_contiguous(dt) || !ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_is_contiguous(mm_b) ||
        (sig && !ggml_is_contiguous(sig)) || (sig && ggml_nelements(sig) != nh*nt) ||
        ggml_nelements(mul) != nh*nt || add->ne[0] != nh || (add->ne[1] != nt && !flat_slots2)) {
        COLA_FAIL();
    }
    if (!ggml_cuda_cola_gates_supported(mm_a->src[0], mm_a->src[1]) || !ggml_cuda_cola_gates_supported(mm_b->src[0], mm_b->src[1])) {
        COLA_FAIL();
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool mmvf_a = ggml_cuda_should_use_mmvf(mm_a->src[0], cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, mm_a->src[1]->ne[1]);
    const bool mmvf_b = ggml_cuda_should_use_mmvf(mm_b->src[0], cc, ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, mm_b->src[1]->ne[1]);
    // on sm_86 the projections of the decode batch with n_tokens > 1 do NOT go through mul_mat_vec_f (that dispatch
    // requires ne11 == 1 when ampere MMA is available) — they fall to the tensor-core/cuBLAS GEMM. GGML_COLA_GATES_ANY=1
    // fuses regardless, =2 fuses only where the original dispatch is NOT mul_mat_vec_f (the GEMM that costs 9,8% do passo).
    const int any = ggml_cuda_cola_gates_any();
    if (any == 0 ? (!mmvf_a || !mmvf_b) : (any == 2 && (mmvf_a || mmvf_b))) {
        COLA_FAIL();
    }
    // sub-graph: every node in [i, last] (views included); only the gate and the beta result (sigmoid, or the raw projection) leave it
    const int count = last - i + 1;
    if (count > 24) {
        COLA_FAIL();
    }
    int idxs[24];
    ggml_op ops[24];
    const int outs[2] = { seq[3], sig ? seq[5] : seq[4] };
    int nl = 0;
    for (int k = 0; k < count; ++k) {
        const ggml_tensor * t = cgraph->nodes[i + k];
        // views of the results stay outside the sub-graph (they are consumed by later nodes)
        if (t->view_src != nullptr && (t->view_src == cgraph->nodes[outs[0]] || t->view_src == cgraph->nodes[outs[1]])) {
            continue;
        }
        idxs[nl] = i + k;
        ops[nl]  = t->op;
        nl++;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, nl, ops, outs, 2)) {
        COLA_FAIL();
    }
    {
        // the results may not overlap dt / a nor each other. The sigmoid result may sit on the activations (SIGK parks it in a scratch
        // buffer until every block is done reading them); the gate and the raw projection may not.
        const ggml_tensor * y   = mm_a->src[1];
        const ggml_tensor * res = sig ? sig : mm_b;
        bool ok = ggml_cuda_sk_disjoint(mul, res) && ggml_cuda_sk_disjoint(mul, y) && ggml_cuda_sk_disjoint(mul, dt) && ggml_cuda_sk_disjoint(mul, a) &&
                  ggml_cuda_sk_disjoint(res, dt) && ggml_cuda_sk_disjoint(res, a);
        if (!sig) {
            ok = ok && ggml_cuda_sk_disjoint(res, y);
        }
        if (!ok) {
            COLA_FAIL();
        }
    }
    ggml_cuda_cola_gdn_gates(*ctx, mm_a, mm_b, dt, a, mul, sig);
    return count - 1;
}

static int ggml_cuda_ikcs_try_fuse(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const bool ikcs_off = !(ggml_cuda_cola_mask() & GGML_COLA_CONV);
    if (ikcs_off) {
        return 0;
    }

    const ggml_tensor * cat = cgraph->nodes[i];
    if (cat->op != GGML_OP_CONCAT || cat->type != GGML_TYPE_F32 || (cat->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_get_op_params_i32(cat, 0) != 0 || !ggml_is_contiguous(cat) || cat->ne[3] != 1) {
        return 0;
    }

    const ggml_tensor * st  = cat->src[0];  // [3, C, n_s]
    const ggml_tensor * qkv = cat->src[1];  // [n_t, C, n_s]
    if (!st || !qkv || st->type != GGML_TYPE_F32 || qkv->type != GGML_TYPE_F32) {
        return 0;
    }
    const int64_t C   = cat->ne[1];
    const int64_t n_s = cat->ne[2];
    const int64_t n_t = qkv->ne[0];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != n_s || st->ne[3] != 1 ||
        qkv->ne[1] != C || qkv->ne[2] != n_s || qkv->ne[3] != 1 || n_t < 1 || n_t > GGML_IKCS_MAX_T ||
        cat->ne[0] != 3 + n_t || n_s > 65535) {
        return 0;
    }
    const size_t f = sizeof(float);
    if (st->nb[0] != f || st->nb[1] % f || st->nb[2] % f || qkv->nb[0] % f || qkv->nb[1] % f || qkv->nb[2] % f ||
        !st->data || !qkv->data) {
        return 0;
    }

    // walk the chain: (VIEW, CPY) x K, SSM_CONV, SILU
    const ggml_tensor * cpys[GGML_IKCS_MAX_SNAP];
    const ggml_tensor * views[GGML_IKCS_MAX_SNAP];
    int                 view_idx[GGML_IKCS_MAX_SNAP];
    int                 n_views = 0;
    int                 n_snap  = 0;
    int                 ssm_idx = -1;
    int                 act_idx = -1;
    const ggml_tensor * ssm     = nullptr;
    const ggml_tensor * act     = nullptr;

    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            if (n->op == GGML_OP_VIEW && n->view_src == cat) {
                if (n_views == GGML_IKCS_MAX_SNAP) {
                    return 0;
                }
                views[n_views]    = n;
                view_idx[n_views] = j;
                ++n_views;
            }
            continue;
        }
        if (ssm == nullptr) {
            if (n->op == GGML_OP_CPY && n->src[0] && n->src[0]->op == GGML_OP_VIEW && n->src[0]->view_src == cat &&
                !(n->flags & GGML_TENSOR_FLAG_OUTPUT) && n_snap < GGML_IKCS_MAX_SNAP) {
                cpys[n_snap++] = n;
                continue;
            }
            if (n->op == GGML_OP_SSM_CONV && n->src[0] == cat && !(n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                ssm     = n;
                ssm_idx = j;
                continue;
            }
            return 0;
        }
        if (n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU && n->src[0] == ssm) {
            act     = n;
            act_idx = j;
        }
        break;
    }
    if (!ssm || !act || n_views != n_snap) {
        return 0;
    }

    // uses: concat -> {K views, ssm_conv}; each view -> one cpy; ssm_conv -> silu only
    if (ggml_node_get_use_count(cgraph, i) != n_snap + 1 || !ggml_node_has_n_uses(cgraph, ssm_idx, 1)) {
        return 0;
    }
    for (int k = 0; k < n_snap; ++k) {
        const ggml_tensor * v = cpys[k]->src[0];
        int vi = -1;
        for (int q = 0; q < n_views; ++q) {
            if (views[q] == v) {
                vi = view_idx[q];
            }
        }
        if (vi < 0 || ggml_node_get_use_count(cgraph, vi) != 1) {
            return 0;
        }
    }

    // ssm_conv / silu: d_conv 4, F32, channel-contiguous
    const ggml_tensor * w = ssm->src[1];
    if (!w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || w->nb[0] != f || w->nb[1] % f || !w->data ||
        ssm->type != GGML_TYPE_F32 || ssm->ne[0] != C || ssm->ne[1] != n_t || ssm->ne[2] != n_s ||
        act->type != GGML_TYPE_F32 || act->ne[0] != C || act->ne[1] != n_t || act->ne[2] != n_s || act->ne[3] != 1 ||
        act->nb[0] != f || act->nb[1] % f || act->nb[2] % f || !act->data) {
        return 0;
    }

    // snapshots: src = [3, C, n_s] window of the concat, dst = [3*C, n_s] rows of the conv cache
    for (int k = 0; k < n_snap; ++k) {
        const ggml_tensor * v = cpys[k]->src[0];
        const ggml_tensor * d = cpys[k]->src[1];
        if (v->type != GGML_TYPE_F32 || v->ne[0] != 3 || v->ne[1] != C || v->ne[2] != n_s || v->ne[3] != 1 ||
            v->nb[0] != f || v->nb[1] != cat->nb[1] || v->nb[2] != cat->nb[2] ||
            v->view_offs % f != 0 || v->view_offs / f > (size_t) n_t) {
            return 0;
        }
        if (!d || d->type != GGML_TYPE_F32 || !d->data || d->ne[0] != 3 * C || d->ne[1] != n_s || d->ne[2] != 1 ||
            d->ne[3] != 1 || d->nb[0] != f || d->nb[1] % f) {
            return 0;
        }
    }

    // memory: the fused kernel writes act + the snapshots while the state/x/weights are the only inputs.
    // act may alias x element for element (all of a thread's inputs are loaded before its first store); any other
    // overlap, and any overlap of a snapshot with an input, disables the fusion.
    const auto range = [](const ggml_tensor * t, int64_t & b, int64_t & e) {
        b = (int64_t) t->data;
        e = b + (int64_t) ggml_backend_buft_get_alloc_size(t->buffer->buft, t);
    };
    const auto overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        int64_t a0, a1, b0, b1;
        range(a, a0, a1);
        range(b, b0, b1);
        return a0 < b1 && b0 < a1;
    };
    // se o alocador colocou a saida do conv (act) em cima de um dos insumos (st, w ou qkv),
    // o kernel fundido escreve num scratch do backend e UM cpy leva o resultado para act. A
    // aritmetica e' a mesma; muda so' o endereco de escrita. GGML_IKCS_TMP=0 volta ao comportamento
    // conservador (recusa), que e' o A/B do artefato.
    static const bool ikcs_tmp_enabled = getenv("GGML_IKCS_TMP") == nullptr || atoi(getenv("GGML_IKCS_TMP")) != 0;
    bool ikcs_tmp = false;
    if (overlap(act, st) || overlap(act, w)) {
        if (!ikcs_tmp_enabled || !ggml_is_contiguous(act)) {
            return 0;
        }
        ikcs_tmp = true;
    }
    if (overlap(act, qkv)) {
        // x^T is a transposed view of the projection: x(t, c, s) at data + t*nb0 + c*nb1 + s*nb2
        const bool same_map = act->data == qkv->data && act->nb[0] == qkv->nb[1] && act->nb[1] == qkv->nb[0] &&
                              (n_s == 1 || act->nb[2] == qkv->nb[2]);
        if (!same_map) {
            if (!ikcs_tmp_enabled || !ggml_is_contiguous(act)) {
                return 0;
            }
            ikcs_tmp = true;
        }
    }
    for (int k = 0; k < n_snap; ++k) {
        const ggml_tensor * d = cpys[k]->src[1];
        if (overlap(d, st) || overlap(d, qkv) || overlap(d, w) || overlap(d, act)) {
            return 0;
        }
        for (int q = 0; q < k; ++q) {
            if (overlap(d, cpys[q]->src[1])) {
                return 0;
            }
        }
    }

    if (ggml_cuda_cola_log()) fprintf(stderr, "[ikcs] fused nt=%lld ns=%lld snapshots=%d scratch=%d\n", (long long)n_t, (long long)n_s, n_snap, ikcs_tmp);
    ggml_cuda_ikcs_launch(*cuda_ctx, cat, cpys, n_snap, ssm, act, ikcs_tmp);
    return act_idx - i;
}

#undef COLA_FAIL
#endif
