/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/reorder/ir_builder.hpp"
#include "gpu/intel/jit/reorder/normalization.hpp"
#include "gpu/intel/jit/reorder/tiler.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/ir/epilogue.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

void reorder_ir_builder_t::build() {
    const auto &tiles = cfg_.tiles();
    const auto &thr_tile = tiles.front();

    for (const auto &iter_tile : tiles) {
        if (try_build(thr_tile, iter_tile)) {
            auto tg_dims = thr_tile.dims();
            auto tg_dim = cfg_.tg_dim();
            if (tg_dim < thr_tile.ndims())
                tg_dims[tg_dim] *= reorder_config_t::tg_factor;
            tensor_t tg_tile(tg_dims);

            gpu_info() << "Reorder configuration:";
            gpu_info() << "  Source layout:            " << src_layout_;
            gpu_info() << "  Destination layout:       " << dst_layout_;
            gpu_info() << "  Iteration tile:           " << thr_tile.str();
            gpu_info() << "  Loop tile:                " << iter_tile.str();
            gpu_info() << "  Thread group tile:        " << tg_tile.str();
            return;
        }
    }
    gpu_error_not_expected();
}

bool reorder_ir_builder_t::try_build(
        const tensor_t &thr_tile, const tensor_t &iter_tile) {
    constraint_set_t init_cset;

    dim_idx_t ndims = src_layout_.ndims();
    std::vector<expr_t> vars;
    for (dim_idx_t i = 0; i < ndims; i++) {
        char letter = dim_idx::as_tag(i);
        vars.push_back(var_t::make(type_t::s32(), std::string(1, letter)));
    }

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(cfg_.kernel_grid(), cfg_.thread_group_grid(),
            cfg_.exec_cfg().simd(), init_cset, init_stmts);

    const auto &vdims = cfg_.vdims();
    std::unordered_map<std::string, dim_t> vdim_map;
    for (dim_idx_t i = 0; i < ndims; i++) {
        vdim_map[vars[i].as<var_t>().name] = vdims[i];
    }

    view_t src_view(vars, ndims);
    for (dim_idx_t i = 0; i < ndims; i++) {
        src_view.set_vdim(vars[i], vdims[i]);
        src_view.set_tdim(i, vars[i]);
    }
    src_view.set_tlayout(src_layout_);
    src_view.set_tmasks(vdim_map);

    view_t dst_view(vars, ndims);
    for (dim_idx_t i = 0; i < ndims; i++) {
        dst_view.set_vdim(vars[i], vdims[i]);
        dst_view.set_tdim(i, vars[i]);
    }
    dst_view.set_tlayout(dst_layout_);
    dst_view.set_tmasks(vdim_map);

    gemm_schedule_t schedule(
            init_cset, cfg_.kernel_grid(), cfg_.thread_group_grid());

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    const auto &grid_map = cfg_.grid_map();
    std::array<std::vector<expr_t>, 3> fused_idxs;
    for (dim_idx_t i = 0; i < ndims; i++) {
        std::vector<expr_t> ordered;
        auto v = vars[i];
        if (iter_tile(i) != 1) {
            expr_t outer, inner;
            schedule.split(v, iter_tile(i), outer, inner);
            schedule.tensorize(inner);
            v = outer;
            ordered.insert(ordered.begin(), outer);
        }
        if (iter_tile(i) != thr_tile(i)) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            dim_t inner_bound = thr_tile(i) / iter_tile(i);
            schedule.split(v, inner_bound, outer, inner);
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        if (i == cfg_.tg_dim()) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, reorder_config_t::tg_factor, outer, inner);
            schedule.bind(inner, cfg_.thread_group_grid().idx(grid_map[i]));
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        fused_idxs[grid_map[i]].push_back(std::move(v));
        schedule.reorder(ordered);
    }

    for (int i = 0; i < (int)fused_idxs.size(); i++) {
        auto &vec = fused_idxs[i];
        if (vec.empty()) continue;
        auto var = (vec.size() == 1 ? vec[0] : schedule.fuse(vec));
        schedule.bind(var, cfg_.kernel_grid().idx(i));
    }

    schedule.finalize();

    auto thr_view_tile
            = schedule.thr_view_tile(src_view, /*is_relative=*/false);

    auto src_thr_view = src_view.create_sub_view(thr_view_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_view_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    ir_context_t ir_ctx(cfg_.exec_cfg(), init_cset);
    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read_params = get_send_params(cfg_.exec_cfg(), send_op_t::load,
            send_address_t::a64, src_thr_view, true);
    read_params.try_legacy = false;
    auto read = make_access_builder(
            ir_ctx, src_thr_view, src_buf, reg_buf, read_params);
    auto &read_stmt = read.stmt();

    auto write_params = get_send_params(cfg_.exec_cfg(), send_op_t::store,
            send_address_t::a64, dst_thr_view, true);
    write_params.try_legacy = false;
    auto write = make_access_builder(
            ir_ctx, dst_thr_view, dst_buf, reg_buf, write_params);
    auto write_stmt = write.stmt();

    auto &read_layout = read.reg_layout();
    auto &write_layout = write.reg_layout();
    int read_buf_size = read.reg_buf_size();
    int write_buf_size = write.reg_buf_size();

    bool has_post_ops = dst_md_ && attr_
            && (!attr_->post_ops_.has_default_values()
                    || !attr_->zero_points_.has_default_values()
                    || !attr_->scales_.has_default_values()
                    || !attr_->rounding_mode_.has_default_values());

    if (has_post_ops) {
        post_op_view_mapper_t view_mapper(dst_view);
        post_op_context_t post_op_ctx(*attr_, cfg_.zp_cfg(), schedule,
                kernel_info_, *dst_md_, *dst_md_, view_mapper);
        write_stmt = create_epilogue_stmt(cfg_.exec_cfg(), ir_ctx, schedule,
                /*force_c_reorder=*/true, post_op_ctx, thr_view_tile,
                read_layout, dst_buf, reg_buf, write_buf_size);
    } else if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(
                alloc_t::make(tmp_buf, write_buf_size, alloc_kind_t::grf));
        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    } else {
        read_buf_size = std::max(read_buf_size, write_buf_size);
    }

    allocs.push_back(alloc_t::make(reg_buf, read_buf_size, alloc_kind_t::grf));

    stmt_ = stmt_t();
    stmt_ = stmt_.append(read_stmt);
    stmt_ = stmt_.append(write_stmt);

    stmt_ = schedule.create_loop_nest(stmt_);
    stmt_ = schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    stmt_ = inject_external_var_let(stmt_, ir_ctx);

    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = fix_int32_overflow(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(
            stmt_, ir_ctx, cfg_.exec_cfg().regs() * cfg_.exec_cfg().grf_size());
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    int ir_regs = get_peak_regs(stmt_, cfg_.exec_cfg().grf_size());
    int reserved_regs = 16;
    int regs = ir_regs + reserved_regs;
    if (regs > cfg_.exec_cfg().regs()) {
        gpu_warning() << "Estimated GRF usage is " << regs
                      << " registers which exceeds available space, retry with "
                         "a smaller tile.";

        return false;
    }

    gpu_trace() << "Reorder kernel body:\n" << stmt_;
    return true;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
