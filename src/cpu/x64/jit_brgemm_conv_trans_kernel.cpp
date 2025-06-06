/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

namespace jit_avx512_core_brgemm_conv_trans_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_trans_kernel_args_t, field)

jit_avx512_core_brgemm_conv_trans_kernel_t::
        jit_avx512_core_brgemm_conv_trans_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp, const char *name)
    : jit_generator_t(name)
    , jcp(ajcp)
    , inp_dsz(jcp.src_dsz)
    , ic_block_sz(
              jcp.is_reduced_rtus ? jcp.rtus_padded_ic_size : jcp.inp_ic_block)
    , ic_block_offset(inp_dsz * ic_block_sz)
    , iw_size(inp_dsz * jcp.ngroups * jcp.ic_without_padding)
    , dst_w_block(dst_w(jcp, jcp.ow_block))
    , dst_stride(jcp.copy_block_only ? dst_w_block : jcp.iwp)
    , VL(cpu_isa_traits_t<avx512_core>::vlen)
    , n_vec(ic_block_sz / jcp.simd_w)
    , n_tail_vec((jcp.ic_without_padding % ic_block_sz) / jcp.simd_w) {

    const auto kh_stride
            = (jcp.relo_type == conv_brgemm_relo_type_t::whi) ? jcp.kh : 1;
    dst_w_offset = kh_stride * ic_block_offset;
    dst_h_offset = dst_stride * dst_w_offset;
}

int get_inp_size(int dst_size, int ext_k, int stride, int dilate) {
    const auto res = calculate_end_padding(0, dst_size, 0, stride, ext_k);
    return res;
}

int get_inp_start(int b, int b_size, int stride, int pad) {
    return b * b_size * stride - pad;
}

int jit_avx512_core_brgemm_conv_trans_kernel_t::inp_w(int out_w, int kw) const {
    return get_inp_size(out_w, kw, jcp.stride_w, jcp.dilate_w);
}

int jit_avx512_core_brgemm_conv_trans_kernel_t::inp_w(int out_w) const {
    return inp_w(out_w, jcp.ext_kw);
}

int jit_avx512_core_brgemm_conv_trans_kernel_t::dst_w(
        const jit_brgemm_conv_conf_t &ajcp, int out_w) {
    int res = 0;
    res = get_inp_size(out_w, ajcp.ext_kw, ajcp.stride_w, ajcp.dilate_w);
    if (ajcp.is_os_blocking) res = rnd_up(res, ajcp.stride_w);
    return res;
}

int jit_avx512_core_brgemm_conv_trans_kernel_t::inp_w_start(int owb) const {
    return get_inp_start(owb, jcp.ow_block, jcp.stride_w, jcp.l_pad);
}

// use different vmovdqu32/16/8 due to case when tail mask used
void jit_avx512_core_brgemm_conv_trans_kernel_t::load(
        const Xbyak::Xmm &x, const Xbyak::Address &addr) {
    if (one_of(jcp.src_dt, f32, s32))
        vmovdqu32(x, addr);
    else if (one_of(jcp.src_dt, bf16, f16))
        vmovdqu16(x, addr);
    else if (one_of(jcp.src_dt, s8, u8, f8_e5m2, f8_e4m3))
        vmovdqu8(x, addr);
    else
        assert(!"Unknown type!");
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::store(
        const Xbyak::Address &addr, const Xbyak::Xmm &x) {
    if (one_of(jcp.src_dt, f32, s32))
        vmovdqu32(addr, x);
    else if (one_of(jcp.src_dt, bf16, f16))
        vmovdqu16(addr, x);
    else if (one_of(jcp.src_dt, s8, u8, f8_e5m2, f8_e4m3))
        vmovdqu8(addr, x);
    else
        assert(!"Unknown type!");
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::zero_ic_block(
        bool is_ic_tail, dim_t dst_off) {
    bool has_block_tail = (jcp.inp_ic_block % jcp.simd_w);
    auto zmm = jcp.s8s8_compensation_required ? zmm_s8s8_shift : zmm_zero;

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++)
        store(ptr[aux_dst_ptr + dst_off + iv * VL], zmm);
    const auto last_dst_off = aux_dst_ptr + dst_off + nvec * VL;
    if (is_ic_tail) {
        if (has_block_tail)
            store(ptr[last_dst_off] | kblock_tail_mask | T_z, zmm);
        else
            store(ptr[last_dst_off], zmm);
    } else if (has_block_tail)
        store(ptr[last_dst_off] | kblock_tail_mask | T_z, zmm);
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_ic_block(dim_t zidx,
        bool is_ic_tail, dim_t inp_off = 0, dim_t dst_off = 0,
        bool do_load = true) {
    bool has_block_tail = (jcp.inp_ic_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    Xbyak::Zmm zmm_tmp = get_zmm(zidx);
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++) {
        if (do_load) load(zmm_tmp, ptr[aux_inp_ptr + inp_off + iv * VL]);
        store(ptr[aux_dst_ptr + dst_off + iv * VL], zmm_tmp);
    }
    const auto last_inp_off = aux_inp_ptr + inp_off + nvec * VL;
    const auto last_dst_off = aux_dst_ptr + dst_off + nvec * VL;

    if (is_ic_tail) {
        auto zmm_tmp_mask = zmm_tmp | ktail_mask | T_z;
        if (do_load) load(zmm_tmp_mask, ptr[last_inp_off]);
        if (has_block_tail)
            store(ptr[last_dst_off] | kblock_tail_mask | T_z, zmm_tmp);
        else
            store(ptr[last_dst_off], zmm_tmp);
    } else if (has_block_tail) {
        auto zmm_tmp_mask = zmm_tmp | kblock_tail_mask | T_z;
        if (do_load) load(zmm_tmp_mask, ptr[last_inp_off]);
        store(ptr[last_dst_off] | kblock_tail_mask | T_z, zmm_tmp);
    }
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::generate() {
    preamble();

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(dst_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(reg_hc, ptr[param1 + GET_OFF(h_count)]);
    mov(reg_t_pad, ptr[param1 + GET_OFF(t_pad)]);
    mov(reg_b_pad, ptr[param1 + GET_OFF(b_pad)]);
    mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
    mov(reg_ic, ptr[param1 + GET_OFF(ic)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    if (jcp.ic_without_padding % jcp.inp_ic_block) {
        int tail_size
                = (jcp.ic_without_padding % jcp.inp_ic_block) % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    if (jcp.inp_ic_block % jcp.simd_w) {
        int block_tail_size = jcp.inp_ic_block % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << block_tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(kblock_tail_mask, reg_tmp);
    }
    if (jcp.s8s8_compensation_required) {
        mov(reg_tmp, -128);
        vpbroadcastb(zmm_s8s8_shift, reg_tmp.cvt8());
    }

    auto icb_loop_body = [&](bool is_ic_tail) {
        Xbyak::Label kh_label, no_kh_label;
        Xbyak::Label finish_label;

        mov(aux_inp_ptr, inp_ptr);
        mov(aux_dst_ptr, dst_ptr);

        cmp(reg_hc, 0);
        jle(finish_label, T_NEAR); // nothing to do

        if (jcp.t_pad > 0) {
            Xbyak::Label kh_tover_label, no_kh_tover_label;
            cmp(reg_t_pad, 0);
            jle(no_kh_tover_label, T_NEAR);

            mov(kh_over, reg_t_pad);
            L(kh_tover_label);
            {
                // TODO: adjust step to improve zeroing efficiency for small ic
                for (dim_t iw = 0; iw < dst_w_block; iw++)
                    zero_ic_block(is_ic_tail, iw * dst_w_offset);
                add(aux_dst_ptr, dst_h_offset);

                dec(kh_over);
                jnz(kh_tover_label, T_NEAR);
            }
            sub(reg_hc, reg_t_pad);
            L(no_kh_tover_label);
        }

        cmp(reg_hc, reg_b_pad);
        jle(no_kh_label, T_NEAR);

        L(kh_label);
        {
            copy_ow_block(is_ic_tail);
            auto inp_h_offset = jcp.iw * iw_size;

            add(aux_inp_ptr, inp_h_offset);
            add(aux_dst_ptr, dst_h_offset);

            dec(reg_hc);
            cmp(reg_hc, reg_b_pad);
            jg(kh_label, T_NEAR);
        }
        L(no_kh_label);

        {
            // even if jcp.b_pad == 0 it is possible that b_pad passed as
            // parameter is not equal to 0 (e.g. for cases with tail by oh)
            Xbyak::Label kh_bover_label, no_kh_bover_label;
            cmp(reg_hc, 0);
            jle(no_kh_bover_label, T_NEAR);

            L(kh_bover_label);
            {
                // TODO: adjust step to improve zeroing efficiency for small ic
                for (dim_t iw = 0; iw < dst_w_block; iw++)
                    zero_ic_block(is_ic_tail, iw * dst_w_offset);
                add(aux_dst_ptr, dst_h_offset);

                dec(reg_hc);
                jnz(kh_bover_label, T_NEAR);
            }
            L(no_kh_bover_label);
        }

        L(finish_label);

        // End IC Loop
        auto inp_cb_offset = ic_block_offset;
        auto dst_cb_offset = jcp.ihp * dst_h_offset;

        add(inp_ptr, inp_cb_offset);
        add(dst_ptr, dst_cb_offset);
    };

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
        Xbyak::Label ic_tail_label, icb_continue_label;
        add(reg_ic, jcp.inp_ic_block);
        cmp(reg_ic, jcp.ic);
        jg(ic_tail_label, T_NEAR);

        icb_loop_body(false);
        jmp(icb_continue_label, T_NEAR);

        L(ic_tail_label);
        icb_loop_body(true);

        L(icb_continue_label);
    }

    postamble();
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_ow_block(
        bool is_ic_tail) {
    if (jcp.nb_ow == 1) {
        copy_ow_block_body(jcp.l_pad, jcp.ow_block, jcp.iw, is_ic_tail);
        return;
    }

    Xbyak::Label copy_block_done_label;

    int start_first_zero_block = -1;
    int end_first_zero_block = -1;
    int start_first_partial_block = -1;
    int end_first_partial_block = -1;
    int start_full_block = -1;
    int end_full_block = -1;
    int start_last_partial_block = -1;
    int end_last_partial_block = -1;

    const auto adj_iw = nstl::min(jcp.iw, jcp.iwp - jcp.l_pad);

    int ow_block_tail = jcp.ow % jcp.ow_block;

    for (int owb = 0; owb < jcp.nb_ow; owb++) {
        const auto inp_block = inp_w(jcp.ow_block);
        const auto inp_start = inp_w_start(owb);
        const auto inp_end = inp_start + inp_block;
        if (inp_start + inp_block < 0) {
            if (start_first_zero_block == -1) start_first_zero_block = owb;
            end_first_zero_block = owb;
        } else if (inp_start < 0) {
            if (start_first_partial_block == -1)
                start_first_partial_block = owb;
            end_first_partial_block = owb;
        } else if (inp_start < adj_iw) {
            if (inp_end <= adj_iw) {
                if (start_full_block == -1) start_full_block = owb;
                end_full_block = owb;
            } else {
                if (start_last_partial_block == -1)
                    start_last_partial_block = owb;
                end_last_partial_block = owb;
            }
        }
    }

    if (start_first_zero_block != -1) {
        Xbyak::Label skip_first_zero_blocks;
        cmp(reg_owb, end_first_zero_block);
        jg(skip_first_zero_blocks, T_NEAR);
        // zero block
        copy_ow_block_body(0, jcp.ow_block, 0, is_ic_tail);
        jmp(copy_block_done_label, T_NEAR);

        L(skip_first_zero_blocks);
    }
    if (start_first_partial_block != -1) {
        for (int b = start_first_partial_block; b <= end_first_partial_block;
                b++) {
            int cur_ow_block = (b == jcp.nb_ow - 1 && ow_block_tail > 0)
                    ? ow_block_tail
                    : jcp.ow_block;
            const auto inp_block = inp_w(cur_ow_block);
            const auto inp_start = inp_w_start(b);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = -inp_start;
            const auto block_len = nstl::min(adj_iw, inp_end);
            Xbyak::Label skip_first_partial_block;
            cmp(reg_owb, b);
            jne(skip_first_partial_block, T_NEAR);
            copy_ow_block_body(block_lpad, jcp.ow_block, block_len, is_ic_tail);
            jmp(copy_block_done_label, T_NEAR);
            L(skip_first_partial_block);
        }
    }
    if (start_full_block != -1) {
        Xbyak::Label skip_full_blocks;
        cmp(reg_owb, end_full_block);
        jg(skip_full_blocks, T_NEAR);
        copy_ow_block_body(0, jcp.ow_block, inp_w(jcp.ow_block), is_ic_tail);
        jmp(copy_block_done_label, T_NEAR);

        L(skip_full_blocks);
    }
    if (start_last_partial_block != -1) {
        for (int b = start_last_partial_block; b <= end_last_partial_block;
                b++) {
            int cur_ow_block = (b == jcp.nb_ow - 1 && ow_block_tail > 0)
                    ? ow_block_tail
                    : jcp.ow_block;
            const auto inp_block = inp_w(cur_ow_block);
            const auto inp_start = inp_w_start(b);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = 0;
            const auto block_len = nstl::min(adj_iw, inp_end) - inp_start;
            Xbyak::Label skip_last_partial_block;
            cmp(reg_owb, b);
            jne(skip_last_partial_block, T_NEAR);
            copy_ow_block_body(block_lpad, cur_ow_block, block_len, is_ic_tail);
            jmp(copy_block_done_label, T_NEAR);

            L(skip_last_partial_block);
        }
    }

    // if not any above case then owb is among last zero blocks
    // check is this needed and check may it be partial
    copy_ow_block_body(0, jcp.ow_block, 0, is_ic_tail);

    L(copy_block_done_label);
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_ow_block_body(
        int lpad, int ow_len, int iw_len, bool is_ic_tail) {
    const auto dst_width = dst_w(jcp, ow_len);
    for (dim_t ind_w = 0; ind_w < dst_width; ind_w++) {
        auto iw_idx = ind_w - lpad;
        auto dst_off = ind_w * dst_w_offset;
        if (iw_idx < 0 || iw_idx >= iw_len) {
            // left or right padding
            zero_ic_block(is_ic_tail, dst_off);
        } else {
            auto inp_off = iw_idx * iw_size;
            copy_ic_block(ind_w, is_ic_tail, inp_off, dst_off, true);
        }
    }
}

jit_avx512_core_brgemm_conv_rtus_kernel_t::
        jit_avx512_core_brgemm_conv_rtus_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jit_avx512_core_brgemm_conv_trans_kernel_t(ajcp, jit_name()) {
    dst_h_offset = jcp.iwp * ic_block_offset;
}

void jit_avx512_core_brgemm_conv_rtus_kernel_t::generate() {
    preamble();

    const Xbyak::Reg64 &reg_khp = reg_hc;
    const Xbyak::Reg64 &reg_kwp = reg_owb;

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(dst_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(reg_khp, ptr[param1 + GET_OFF(h_count)]);
    mov(reg_kwp, ptr[param1 + GET_OFF(owb)]);

    if (jcp.ic_without_padding % jcp.ic_block) {
        int tail_size = (jcp.ic_without_padding % jcp.ic_block) % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    if (jcp.ic_block % jcp.simd_w) {
        int block_tail_size = jcp.ic_block % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << block_tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(kblock_tail_mask, reg_tmp);
    }

    assert(jcp.nb_ic_blocking == 1 && "TODO: support multi-batch case");

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
        const bool is_ic_tail
                = (icb + 1) * jcp.ic_block > jcp.ic_without_padding;
        mov(aux_inp_ptr, inp_ptr);
        mov(aux_dst_ptr, dst_ptr);

        // Section 1: copy nw spatial elements in a row
        Xbyak::Label label_kwp_begin, label_kwp_end;
        cmp(reg_kwp, 0);
        jle(label_kwp_end, T_NEAR);
        L(label_kwp_begin);
        {
            copy_ic_block(0, is_ic_tail);

            auto inp_w_step = jcp.stride_w * iw_size;
            auto out_w_step = ic_block_offset;
            add(aux_inp_ptr, inp_w_step);
            add(aux_dst_ptr, out_w_step);

            dec(reg_kwp);
            jnz(label_kwp_begin, T_NEAR);
        }
        L(label_kwp_end);

        // Section 2: copy nh whole rows of OW spatial elements
        Xbyak::Label label_khp_begin, label_khp_end;
        cmp(reg_khp, 0);
        jle(label_khp_end, T_NEAR);
        L(label_khp_begin);
        {
            for (int ow = 0; ow < jcp.ow; ow++) {
                auto inp_w_off = ow * jcp.stride_w * iw_size;
                auto out_w_off = ow * ic_block_offset;
                copy_ic_block(0, is_ic_tail, inp_w_off, out_w_off);
            }

            auto inp_h_step = jcp.stride_h * jcp.iw * iw_size;
            auto out_h_step = jcp.ow * ic_block_offset;
            add(aux_inp_ptr, inp_h_step);
            add(aux_dst_ptr, out_h_step);

            dec(reg_khp);
            jnz(label_khp_begin, T_NEAR);
        }
        L(label_khp_end);
    }

    postamble();
}

} // namespace jit_avx512_core_brgemm_conv_trans_kernel

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
