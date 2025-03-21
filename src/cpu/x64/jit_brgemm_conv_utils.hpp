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

#ifndef CPU_X64_JIT_BRGEMM_CONV_UTILS_HPP
#define CPU_X64_JIT_BRGEMM_CONV_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_convolution_utils {

constexpr size_t P4K = 4096;

bool is_amx(cpu_isa_t isa);
bool uses_batch_elements(
        brgemm_batch_kind_t brg_type, conv_brgemm_exec_type_t exec_type);

status_t init_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

status_t init_1x1_conf(jit_brgemm_conv_conf_t &jcp, cpu_isa_t isa,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

void set_amx_wsp_per_thread(jit_brgemm_conv_conf_t &jcp);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp);

status_t init_conf_bwd_w(jit_brgemm_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
        memory_desc_t &diff_dst_md, primitive_attr_t &attr, int nthreads);

status_t init_scratchpad_bwd_w(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_conv_conf_t &jcp, memory_desc_t &src_md,
        memory_desc_t &diff_weights_md, memory_desc_t &diff_dst_md);

// TODO: make a part of `jit_brgemm_conv_conf_t` instead?
void get_ow_range(const jit_brgemm_conv_conf_t &jcp, int ow, int kw, int &ow_s,
        int &ow_f);
void get_kw_range(const jit_brgemm_conv_conf_t &jcp, int ow, int &kw_s,
        int &kw_full_s, int &kw_full_f, int &kw_f);

#define BRGEMM_CONV_NDHWGC_ORDER \
    n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow, g, jcp.ngroups, \
            ocb, jcp.nb_oc
#define BRGEMM_CONV_NGCDHW_ORDER \
    n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, odb, jcp.nb_od, ohb, jcp.nb_oh, \
            owb, jcp.nb_ow
#define BRGEMM_CONV_GCNDHW_ORDER \
    g, jcp.ngroups, ocb, jcp.nb_oc, n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh, \
            owb, jcp.nb_ow

#define BRGEMM_CONV_ITERATOR_INIT \
    if (jcp.loop_order == loop_ndhwgc) \
        nd_iterator_init(start, BRGEMM_CONV_NDHWGC_ORDER); \
    else if (jcp.loop_order == loop_ngcdhw) \
        nd_iterator_init(start, BRGEMM_CONV_NGCDHW_ORDER); \
    else if (jcp.loop_order == loop_gcndhw) \
        nd_iterator_init(start, BRGEMM_CONV_GCNDHW_ORDER); \
    else \
        assert(!"Unknown loop order");

#define BRGEMM_CONV_ITERATOR_STEP \
    if (jcp.loop_order == loop_ndhwgc) \
        nd_iterator_step(BRGEMM_CONV_NDHWGC_ORDER); \
    else if (jcp.loop_order == loop_ngcdhw) \
        nd_iterator_step(BRGEMM_CONV_NGCDHW_ORDER); \
    else if (jcp.loop_order == loop_gcndhw) \
        nd_iterator_step(BRGEMM_CONV_GCNDHW_ORDER); \
    else \
        assert(!"Unknown loop order");

} // namespace brgemm_convolution_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
