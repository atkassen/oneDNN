/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/reorder/jit/tiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {
namespace jit {

enum class message_kind_t {
    block,
    scattered,
};

dim_t max_strided_bytes(
        const hw_t &hw, const type_t &src_type, const type_t &dst_type) {
    // These conversions use an additional temporary buffer
    const bool use_smaller_buffer
            = utils::one_of(true, src_type.is_fp8(), dst_type.is_fp8())
            || (src_type.is_x32() && (dst_type.is_bf16() || dst_type.is_f16()))
            || (src_type.is_f16() && dst_type.is_bf16());
    // Assume 12 work registers and the rest are used for buffers
    const int buf_regs = use_smaller_buffer ? 38 : 58;
    //                                        ~^   ^~
    //                            (128 - 12) / 3   (128 - 12) / 2
    // TODO: This should be adjusted when post-ops are present.
    return buf_regs * hw.grf_size();
}

dim_t max_packed_bytes(const hw_t &hw) {
    return 32 * hw.grf_size();
}

dim_t count_block_messages(
        const hw_t &hw, dim_t inner_bytes, dim_t iterations) {
    const auto max_block_owords = hw.grf_size() / 2;
    const auto oword_size = 16;
    const auto owords_per_grf = hw.grf_size() / oword_size;

    dim_t block_owords = max_block_owords / 2;
    auto inner_owords = inner_bytes / oword_size;
    dim_t messages = inner_owords / max_block_owords;
    inner_owords -= messages * max_block_owords;
    // If iterations != 1, tail block messages must end on a grf boundary
    const dim_t lower_bound = iterations == 1 ? 1 : owords_per_grf;
    for (; block_owords >= lower_bound; block_owords >>= 1) {
        if (inner_owords >= block_owords) {
            inner_owords -= block_owords;
            messages++;
        }
    }
    gpu_assert(inner_owords == 0);
    return messages * iterations;
}

dim_t count_scattered_messages(
        const hw_t &hw, dim_t inner_bytes, dim_t iterations, int item_size) {
    constexpr int scattered_message_penalty = 4;
    const int message_items = hw.grf_size() / 2;

    auto inner_items = (iterations * inner_bytes) / item_size;
    auto messages = utils::div_up(inner_items, message_items);
    return messages * scattered_message_penalty;
}

struct message_info_t {
    message_info_t() = default;
    message_info_t(message_kind_t kind, dim_t inner_bytes, dim_t iterations,
            int item_size)
        : kind(kind)
        , inner_bytes(inner_bytes)
        , iterations(iterations)
        , item_size(item_size) {}

    message_kind_t kind = message_kind_t::block;
    dim_t inner_bytes = 0;
    dim_t iterations = 0;
    int item_size = 16;

    dim_t latency(const hw_t &hw) const {
        if (inner_bytes == 0 || iterations == 0) return 0;
        return kind == message_kind_t::block
                ? count_block_messages(hw, inner_bytes, iterations)
                : count_scattered_messages(
                        hw, inner_bytes, iterations, item_size);
    }
};

message_info_t estimate_message_info(
        const hw_t &hw, const layout_t &layout, const tile_t &tile) {
    const auto grf_size = hw.grf_size();
    bool can_use_block_messages = true;
    std::vector<dim_t> outer = tile.values();
    dim_t inner_elems = 1;
    int item_size = 16;

    for (auto &blk : layout.blocks()) {
        auto block = blk.block;
        auto dim = blk.dim;
        if (block == 1) continue;
        if (outer[dim] < block) {
            if (block % outer[dim] == 0) {
                inner_elems *= outer[dim];
                outer[dim] = 1;
            }
            break;
        }

        can_use_block_messages &= (outer[dim] % block == 0);
        inner_elems *= block;
        outer[dim] = utils::div_up(outer[dim], block);
    }

    auto inner_bytes = utils::div_up(layout.type().bitsize() * inner_elems, 8);
    auto iterations = tile_t(outer).elems();
    can_use_block_messages &= (inner_bytes % 16 == 0);
    can_use_block_messages &= (iterations == 1 || inner_bytes % grf_size == 0);

    if (inner_bytes == 0 || iterations == 0) return {};

    auto message_kind = can_use_block_messages ? message_kind_t::block
                                               : message_kind_t::scattered;
    if (!can_use_block_messages)
        // Find the largest unit size we can use
        for (item_size = 8; item_size > 1; item_size >>= 1) {
            if (inner_bytes % item_size == 0) break;
        }
    return {message_kind, inner_bytes, iterations, item_size};
}

// Extended layout block
// The additional field `real_stride` is used to determine a block's position
// in being added to the tile.
struct ext_block_t : layout_block_t {
    ext_block_t() = default;
    ext_block_t(const layout_block_t &b)
        : layout_block_t(b.dim, b.block, stride_t::unknown())
        , real_stride(b.stride) {}
    ext_block_t(const pvar_t &dim, dim_t block, const stride_t &stride,
            const stride_t &real_stride)
        : layout_block_t(dim, block, stride), real_stride(real_stride) {}
    stride_t real_stride = stride_t::undefined();
};

bool stride_less_than(const stride_t &l, const stride_t &r) {
    // N.B.: unknown is interpreted as "larger than any fixed value"
    if (l.is_unknown()) return false; // Both or just l is unknown
    if (r.is_unknown()) return true; // Just r is unknown
    return (dim_t)l < (dim_t)r; // Propagate errors from casting invalid values
}

using blocks_iterator_t = typename std::vector<ext_block_t>::iterator;

std::vector<ext_block_t> get_tile_blocks(std::vector<blocks_iterator_t> &its,
        const std::vector<blocks_iterator_t> &ends) {
    // Idea: For a given dimension, create a sequence of blocks that will be
    // used to construct tiles of incrementally increasing tiles.
    const auto niters = its.size();
    std::vector<ext_block_t> blocks;
    if (its.empty()) return blocks;

    dim_t stride = 1;
    stride_t best_outer_stride = stride_t::unknown();
    pvar_t dim = its[0]->dim;

    auto get_factor = [](dim_t &n) {
        for (dim_t f = 2; f <= n / 2; ++f) {
            if (n % f) continue;
            n /= f;
            return f;
        }
        dim_t f = n;
        n = 1;
        return f;
    };

    while (true) {
        for (size_t i = 0; i < niters; ++i) {
            if (its[i] == ends[i]) return blocks;
        }
        dim_t inner = 0, outer = 0;
        stride_t real_stride = stride_t::unknown();
        for (auto &it : its) {
            if (it->stride.is_unknown()) {
                if (stride_less_than(it->real_stride, best_outer_stride))
                    best_outer_stride = it->real_stride;
                continue;
            }
            // Suppose gcd(a1, a2, ..., aN) == 1
            // and gcd(a1 * b1, a2 * b2, ..., aN * bN) = outer.
            // Let gcd(b1, b2, ..., bN) = inner. Then outer % inner == 0.
            outer = math::gcd(outer, (dim_t)it->stride * it->block / stride);
            inner = math::gcd(inner, it->block);
            if (stride_less_than(it->real_stride, real_stride))
                real_stride = it->real_stride;
        }
        if (!inner) break;
        if (outer == 1) {
            for (auto &it : its) {
                if (it->stride.is_unknown()) continue;
                it++;
            }
            continue;
        }
        for (auto &it : its) {
            if (it->stride.is_unknown()) continue;
            dim_t from_block = outer * stride / (dim_t)it->stride;
            it->block /= from_block;
            it->stride *= from_block;
            it->real_stride *= from_block;
            if (it->block == 1) it++;
        }
        // This block must be added to the tile unbroken to avoid divisibility
        // issues.
        const auto indivisible = outer / inner;
        stride *= indivisible;
        // Inner is a block that can be added piecemeal. Remove its smallest
        // factor > 1 and add it to the blocks.
        while (inner > 1) {
            auto factor = get_factor(inner);
            blocks.emplace_back(dim, factor, stride, real_stride);
            stride *= factor;
            real_stride *= factor;
        }
    }
    if (blocks.empty() || !blocks.back().stride.is_unknown())
        blocks.emplace_back(dim, 1, stride_t::unknown(), best_outer_stride);
    return blocks;
}

std::vector<ext_block_t> get_tile_blocks(const std::vector<layout_t> &layouts,
        std::vector<std::vector<ext_block_t>> &blocks) {
    if (layouts.empty()) return {};
    dim_idx_t ndims = layouts.front().ndims();
    for (auto &l : layouts)
        ndims = std::min(l.ndims(), ndims);

    const auto n = layouts.size();
    std::vector<blocks_iterator_t> its(n);
    std::vector<blocks_iterator_t> ends(n);

    std::vector<ext_block_t> tile_blocks;

    for (size_t j = 0; j < n; ++j)
        its[j] = ends[j] = blocks[j].begin();

    for (dim_idx_t i = 0; i < ndims; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const auto end = blocks[j].end();
            while (ends[j] != end && ends[j]->dim.index() == i)
                ++ends[j];
        }

        auto dim_blocks = get_tile_blocks(its, ends);
        tile_blocks.insert(
                tile_blocks.end(), dim_blocks.begin(), dim_blocks.end());

        std::swap(its, ends);
        ends = its;
    }

    auto by_stride = [](const ext_block_t &l, const ext_block_t &r) {
        return stride_less_than(l.real_stride, r.real_stride);
    };

    std::sort(tile_blocks.begin(), tile_blocks.end(), by_stride);
    return tile_blocks;
}

void pad_layouts(std::vector<layout_t> &layouts) {
    if (layouts.empty()) return;
    const auto ndims = layouts.front().ndims();
    std::vector<dim_t> shared_blocks(ndims, 0);
    auto dims = layouts.front().dims();
    uint32_t has_broadcast = 0;

    // Compute the blocking for each dimension that is shared among all layouts
    // which blocking in that dimension (i.e., if the dimension is plain,
    // ignore it).
    auto compute_shared_blocks = [&](const layout_t &l) {
        for (dim_idx_t i = 0; i < ndims; ++i) {
            auto ldim = l.dim(i), &tdim = dims[i];
            has_broadcast |= ((ldim != tdim) << i);
            tdim = std::max(ldim, tdim);
        }
        for (auto &eb : l.enumerated_blocks()) {
            if (!l.is_outermost(eb)) continue;
            auto &b = eb.second;
            auto &inner_block = shared_blocks[b.dim];
            auto inner = l.dim(b.dim) / b.block;
            if (inner == 1) continue;
            inner_block = math::gcd(inner, inner_block);
        }
    };
    for (auto &l : layouts)
        compute_shared_blocks(l);

    // Pad each dimension, including plain dimensions to the shared blocking
    // found above.
    auto pad_layout = [&](layout_t &l) {
        const auto packing = l.type().packing();
        std::vector<layout_block_t> padded_blocks;
        bool seen = false;
        for (auto &eb : l.enumerated_blocks()) {
            padded_blocks.push_back(eb.second);
            auto &b = padded_blocks.back();
            dim_t dim = l.dim(b.dim);
            if (dim == 1) continue;
            if (l.is_outermost(eb)) {
                dim_t block = shared_blocks[b.dim];
                if (!block) block = 8;
                if (!seen && math::gcd(dim, block) % packing) continue;
                dim_t inner = dim / b.block;
                b.block = utils::rnd_up(dim, block) / inner;
            }
            seen = true;
        }
        l = {l.type(), ndims, 0, padded_blocks, /*do_normalize=*/false};
    };
    for (auto &l : layouts)
        pad_layout(l);
}

std::vector<tile_t> generate_tiles(
        const hw_t &hw, std::vector<layout_t> layouts) {
    if (layouts.empty()) return {};
    auto ndims = layouts.front().ndims();
    const bool strict = true;

    auto by_dim = [](const ext_block_t &l, const ext_block_t &r) {
        return l.dim < r.dim;
    };

    auto get_blocks = [&](const layout_t &l) {
        // Partition the layout by dimension. For each dimension, list all of
        // the blocks in order and adjust their strides to be dense. Maintain
        // the original stride as the "real stride" for later sorting purposes.
        auto l_blocks = l.blocks();
        const auto nblocks = l_blocks.size();
        std::vector<ext_block_t> blocks;
        blocks.reserve(nblocks);

        // Find the boundary between the inner blocks and the outer blocks.
        // layout_t::is_outermost will interleave these -- we want to optimize
        // the cases where inner blocks cover the entire dimension, that is,
        // the true outer block is size 1.
        // This will also signal which blocks can be divided arbitrarily in
        // non-strict mode (the true outermost blocks). If tiling in strict
        // mode, tiles must divide dimensions exactly, so skip marking
        // outermost blocks.
        auto inner_idx = nblocks;
        if (!strict) {
            std::vector<bool> seen(l.ndims());
            const auto end = l_blocks.rend();
            for (auto it = l_blocks.rbegin(); it != end; ++it, --inner_idx) {
                if (seen[it->dim] || !inner_idx) break;
                seen[it->dim] = true;
            }
        }

        std::vector<dim_t> strides(l.ndims(), 1);
        for (size_t i = 0; i < nblocks; ++i) {
            auto b = l_blocks[i];
            auto real_stride = b.stride;
            if (i >= inner_idx) {
                b.stride = stride_t::unknown();
                b.block = 1;
            } else
                b.stride = strides[b.dim];
            strides[b.dim] *= b.block;
            blocks.emplace_back(b.dim, b.block, b.stride, real_stride);
        }
        std::sort(blocks.begin(), blocks.end(), by_dim);
        return blocks;
    };

    pad_layouts(layouts);

    // Gather the blocks used for tiling. Also, determine the order of the
    // outermost blocks so that tiles can be expanded along those dimensions
    // in round-robin fashion (non-strict mode only). Finally, determine the
    // maximum tile size required to cover all layouts.
    std::vector<std::vector<ext_block_t>> blocks;
    blocks.reserve(layouts.size());
    std::vector<ext_block_t> outer_blocks(ndims);
    tile_t max_tile = layouts[0].dims();
    for (auto &l : layouts) {
        auto l_blocks = get_blocks(l);
        for (auto &b : l_blocks) {
            if (!b.stride.is_unknown()) continue;
            auto &stride = outer_blocks[b.dim].real_stride;
            if (!stride.is_unknown() || stride_less_than(b.real_stride, stride))
                outer_blocks[b.dim] = b;
        }
        blocks.push_back(std::move(l_blocks));

        for (dim_idx_t i = 0; i < ndims; ++i)
            max_tile[i] = std::min(max_tile[i], l.dim(i));
    }

    // Generate blocks used to incrementally expand the tile to generate a
    // sequence of increasing tiles. When this is exhausted, expand each
    // dimension by a factor of 2 in round-robin fashion according to the order
    // of outermost blocks determined above (non-strict mode only)
    tile_t tile(ndims);
    std::vector<tile_t> rev_tiles = {tile};
    auto tile_blocks = get_tile_blocks(layouts, blocks);
    for (auto &b : tile_blocks) {
        if (b.stride.is_unknown()) continue;
        dim_t stride = b.stride;
        if (stride > tile[b.dim]) {
            gpu_assert(stride % tile[b.dim] == 0);
            tile[b.dim] = stride;
            rev_tiles.push_back(tile);
        }
        tile[b.dim] *= b.block;
        rev_tiles.push_back(tile);
    }

    auto by_stride = [](const ext_block_t &l, const ext_block_t &r) {
        if (l.real_stride.is_undefined()) return false;
        if (r.real_stride.is_undefined()) return true;
        return stride_less_than(l.real_stride, r.real_stride);
    };
    std::sort(outer_blocks.begin(), outer_blocks.end(), by_stride);

    bool have_outer_blocks = true;
    while (have_outer_blocks) {
        have_outer_blocks = false;
        for (auto &block : outer_blocks) {
            if (block.real_stride.is_undefined()) continue;
            if (max_tile[block.dim] <= tile[block.dim]) continue;
            tile[block.dim] *= 2;
            rev_tiles.push_back(tile);
            have_outer_blocks = true;
        }
    }
    return rev_tiles;
}

std::vector<tile_t> tiles(const hw_t &hw, layout_t a, layout_t b) {
    const auto eu_count = hw.eu_count();
    auto cmp = [&](const tile_t &l, const tile_t &r) {
        auto l_threads_reqd = a.elems() / l.elems();
        auto r_threads_reqd = a.elems() / r.elems();
        auto l_eu_util = utils::div_up(l_threads_reqd, eu_count);
        auto r_eu_util = utils::div_up(r_threads_reqd, eu_count);
        auto l_a_msg = estimate_message_info(hw, a, l);
        auto l_b_msg = estimate_message_info(hw, b, l);
        auto r_a_msg = estimate_message_info(hw, a, r);
        auto r_b_msg = estimate_message_info(hw, b, r);
        auto l_msg_load = l_a_msg.latency(hw) + l_b_msg.latency(hw);
        auto r_msg_load = r_a_msg.latency(hw) + r_b_msg.latency(hw);

        // Choose tiles with less message overhead per thread
        if (l_eu_util * l_msg_load != r_eu_util * r_msg_load)
            return (l_eu_util * l_msg_load < r_eu_util * r_msg_load);

        // Choose tiles with more bytes per message
        if (l.elems() * r_msg_load != r.elems() * l_msg_load)
            return (l.elems() * r_msg_load > r.elems() * l_msg_load);

        // If all else fails, go with the bigger tile
        return l.elems() > r.elems();
    };

    auto get_grf_layout_size = [&](const tile_t &tile) {
        auto elems = tile.elems();
        dim_t grf_layout_size = 0;
        for (const auto &l : {a, b}) {
            auto info = estimate_message_info(hw, l, tile);
            int elem_size = std::max(info.item_size, 4);
            int elem_packing = info.item_size / l.type().size();
            auto layout_size = elem_size * elems / elem_packing;
            if (layout_size > grf_layout_size) grf_layout_size = layout_size;
        }
        return grf_layout_size;
    };

    const auto tiles = generate_tiles(hw, {a, b});
    const int elem_size = std::max(a.type().size(), b.type().size());
    const dim_t max_layout_size = max_strided_bytes(hw, a.type(), b.type());
    const dim_t max_elems = max_packed_bytes(hw) / elem_size;

    std::vector<tile_t> candidate_tiles;
    for (auto tile : tiles) {
        if (tile.elems() > max_elems) break;
        if (get_grf_layout_size(tile) > max_layout_size) continue;
        if (candidate_tiles.empty() || tile != candidate_tiles.back())
            candidate_tiles.push_back(std::move(tile));
    }
    gpu_assert(!candidate_tiles.empty());

    size_t best_idx = 0;
    for (size_t i = 0; i < candidate_tiles.size(); ++i)
        if (cmp(candidate_tiles[i], candidate_tiles[best_idx])) best_idx = i;
    candidate_tiles.resize(best_idx + 1);

    return candidate_tiles;
}

} // namespace jit
} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
