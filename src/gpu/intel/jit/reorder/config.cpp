/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/jit/reorder/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

reorder_config_t::reorder_config_t(
        const exec_config_t &ec, layout_t src, layout_t dst) {
    set_exec_cfg(ec);
    const auto &hw = ec.hw();

    reorder::normalize(src, dst);
    src_layout().set_user(src);
    dst_layout().set_user(dst);

    auto dst_elems = utils::rnd_up_pow2(dst.elems());
    auto max_elem_size = std::max(src.type().size(), dst.type().size());
    auto max_elems = std::min(dst_elems, (dim_t)1024 / max_elem_size);
    auto rev_tiles = reorder::tiles(src, dst, max_elems, true);
    auto threads = hw.eu_count() * hw.threads_per_eu();
    auto max_wgs = utils::div_up(threads, ec.simd());
    auto front = rev_tiles.rbegin();
    const auto back = rev_tiles.rend();
    for (; front != back; ++front) {
        auto wgs = utils::div_up(dst_elems, front->elems());
        if (2 * wgs >= max_wgs || front + 1 == back) break;
    }
    tiles_.assign(front, back);

    dim_idx_t ndims = src.ndims();
    tg_tile_idx_ = ndims;
    auto thr_tile = tiles_.front();

    constexpr dim_idx_t grid_ndims = 3;
    grid_map_.resize(ndims, -1);
    vdims_.resize(ndims);
    std::vector<dim_t> kernel_dims(grid_ndims, 1);
    std::vector<dim_t> tg_dims(grid_ndims, 1);
    int grid_idx = 0;

    for (dim_idx_t i = 0; i < ndims; ++i) {
        dim_t tg_dim = thr_tile(i);
        dim_t outer = utils::div_up(dst.dim(i), tg_dim);
        vdims_[i] = outer * tg_dim;
        grid_map_[i] = grid_idx;

        // Heuristic: try to split outer dimension and assign its inner part to
        // the thread group. This may give better bandwidth utilization on
        // XeHP/XeHPG.
        if (tg_tile_idx_ == ndims && outer % tg_factor == 0) {
            outer /= tg_factor;
            tg_dims[grid_idx] *= tg_factor;
            tg_tile_idx_ = i;
        }
        kernel_dims[grid_idx] *= outer;
        if (outer != 1 && grid_idx != grid_ndims - 1) grid_idx++;
    }

    compute::range_t global = compute::range_t::empty(grid_ndims);
    compute::range_t local = compute::range_t::empty(grid_ndims);
    for (dim_idx_t i = 0; i < grid_ndims; ++i) {
        global[i] = kernel_dims[i] * tg_dims[i];
        local[i] = tg_dims[i];
    }
    global[0] *= ec.simd();
    local[0] *= ec.simd();

    nd_range_ = compute::nd_range_t(global, local);
    auto &tg_idxs = ir_builder_t::tg_idxs();
    set_kernel_grid(
            {kernel_dims, std::vector<expr_t>(tg_idxs.begin(), tg_idxs.end())});
    set_thread_group_grid({tg_dims, "thr_idx"});
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
