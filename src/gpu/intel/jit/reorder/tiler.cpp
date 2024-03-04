/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/jit/reorder/tiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

constexpr dim_t runtime_dim() {
    return DNNL_RUNTIME_DIM_VAL;
}

// A checkpoint is a valid dimension for which we can partition a layout.
// `inner` is the shared block amongst all layouts and can be partitioned by
// any divisor. `outer` contains the shared block and a required block that
// cannot be partitioned and must be added first if any block is added to the
// layout partition.
//
// E.g., consider layouts with 24b and 16b innermost inner blocks. The first
// checkpoint for this layout will be {.inner = 8, .outer = 8} because a block
// of 8 is shared between both layouts. One can think of the layouts as 3b8b and
// 2b8b. The next checkpoint (if any exists) will have a block of 2 * 3 = 6 in
// `outer` to account for the incommensurate remaining blocks.

struct checkpoint_t {
    dim_t inner, outer;
};

bool is_fixed_size(const checkpoint_t &cp) {
    return cp.outer != runtime_dim();
}

bool is_fixed_size(const block_t &b) {
    return b.block != runtime_dim();
}

std::vector<checkpoint_t> checkpoints(const std::vector<layout_t> &layouts,
        const std::vector<uint32_t> &masks, dim_idx_t dim_idx,
        bool strict = false) {
    using enumerated_block_t = std::pair<int, block_t>;
    using enumerated_blocks_t = std::vector<enumerated_block_t>;
    using block_iterator_t = typename enumerated_blocks_t::const_iterator;
    std::vector<enumerated_block_t> dim_blocks;
    std::vector<block_t> blocks;
    std::vector<size_t> tail_flags;
    std::vector<block_iterator_t> it;
    std::vector<block_iterator_t> end;

    for (size_t i = 0; i < layouts.size(); ++i) {
        auto &layout = layouts[i];
        auto mask = masks[i];
        // This dimension is broadcast for this layout -- ignore
        if (!(mask & (1 << dim_idx)) && layout.dim(dim_idx) == 1) continue;
        for (auto eb : layout.enumerated_blocks()) {
            auto &block = eb.second;
            if (block.dim_idx != dim_idx) continue;
            if (layout.is_outermost(eb) && !strict) block.block = runtime_dim();
            dim_blocks.push_back(eb);
        }
        tail_flags.push_back(dim_blocks.size());
    }

    size_t offset = 0;
    const auto start = dim_blocks.begin();
    for (auto &f : tail_flags) {
        it.emplace_back(start + offset);
        end.emplace_back(start + f);

        dim_t size = (offset != f) ? it.back()->second.block : 1;
        blocks.emplace_back(dim_idx, size, 1);
        offset = f;
    }

    auto blocks_overlap = [&]() {
        if (blocks.empty()) return true;
        dim_t ref_l = 0, ref_u = std::numeric_limits<dim_t>::max();

        for (const auto &test : blocks) {
            if (!is_fixed_size(test)) continue;
            dim_t test_l = test.stride, test_u = test_l + test.block;
            if (ref_u < test_l || test_u < ref_l) return false;
            ref_l = std::max(ref_l, test_l);
            ref_u = std::min(ref_u, test_u);
        }

        for (const auto &test : blocks) {
            if (is_fixed_size(test)) continue;
            if (ref_u < test.stride) return false;
        }

        return true;
    };

    auto has_next = [&]() {
        for (size_t i = 0; i < it.size(); ++i)
            if (it[i] != end[i]) return true;
        return false;
    };

    auto checkpoint = [&]() -> checkpoint_t {
        dim_t outer = 0, inner = 0;
        for (const auto &b : blocks) {
            if (!is_fixed_size(b)) continue;
            outer = math::gcd(b.block * (dim_t)b.stride, outer);
            inner = math::gcd(b.block, inner);
        }

        if (outer == 0) return {runtime_dim(), runtime_dim()};

        for (const auto &b : blocks)
            if (outer % (dim_t)b.stride) return {1, 1};
        return {inner, outer};
    };

    auto advance = [](block_t &b, block_iterator_t &it,
                           const block_iterator_t &end) {
        b.stride *= b.block;
        b.block = 1;
        if (it == end || ++it == end) return;
        b.block = it->second.block;
    };

    std::vector<checkpoint_t> checkpoints;
    while (has_next()) {
        auto cp = checkpoint();
        if (!is_fixed_size(cp)) {
            checkpoints.push_back(cp);
            break;
        }
        if (blocks_overlap() && cp.outer > 1) {
            bool already_advanced = false;
            for (size_t i = 0; i < blocks.size(); ++i) {
                block_t &b = blocks[i];
                if (!is_fixed_size(b))
                    already_advanced = true;
                else if (b.block == cp.outer) {
                    already_advanced = true;
                    advance(b, it[i], end[i]);
                } else
                    b.block = b.block / cp.outer;
                b.stride = 1;
            }
            checkpoints.push_back(cp);
            if (already_advanced) continue;
        }
        break;

        std::vector<size_t> indices;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (it[i] == end[i]) continue;
            if (indices.empty()
                    || blocks[indices[0]].stride == blocks[i].stride)
                indices.push_back(i);
            else if (blocks[indices[0]].stride > blocks[i].stride) {
                indices.clear();
                indices.push_back(i);
            }
        }
        if (indices.empty()) break;

        for (auto i : indices) {
            block_t &b = blocks[i];
            dim_t stride = b.block * (dim_t)b.stride;
            advance(b, it[i], end[i]);
            b.stride = stride;
        }
    }
    return checkpoints;
}

struct tiling_dim_state_t {
    std::vector<checkpoint_t> checkpoints;
    size_t offset = 0;
    dim_t block = 1;
    dim_t rem_block = 1;

    tiling_dim_state_t() = default;
    tiling_dim_state_t(std::vector<checkpoint_t> checkpoints)
        : checkpoints(std::move(checkpoints)) {
        operator++();
    }

    static dim_t get_factor(dim_t n) {
        for (dim_t i = 2; i < n; ++i)
            if (n % i == 0) return i;
        return n;
    }

    dim_t operator*() const { return block; }

    tiling_dim_state_t &operator++() {
        while (true) {
            if (rem_block > 1) {
                auto factor = get_factor(rem_block);
                block = factor;
                rem_block /= factor;
                return *this;
            }

            if (offset >= checkpoints.size()) {
                block = 0;
                rem_block = 0;
                return *this;
            }

            auto cp = checkpoints[offset];
            if (!is_fixed_size(cp)) {
                block = 2;
                rem_block = 1;
                return *this;
            }

            ++offset;
            rem_block = cp.inner;
            if (cp.outer != cp.inner) {
                block = cp.outer / cp.inner;
                return *this;
            }
        }
    }

    bool has_next() const {
        return offset < checkpoints.size() || block != 0 || rem_block != 0;
    }
};

struct option_t : public block_t {
    option_t() = default;
    option_t(dim_idx_t dim_idx, dim_t block, stride_t stride, type_t type = {})
        : block_t(dim_idx, block, stride), type(type) {}
    type_t type;
};

block_t next_block(const std::vector<layout_t> &layouts,
        const std::vector<dim_t> &dims, dim_t max_block_elems,
        const std::vector<tiling_dim_state_t> &state, bool strict = false) {
    std::vector<option_t> options;
    for (const auto &l : layouts) {
        uint32_t skip_mask = 0;
        auto dims_ = dims;
        // Outermost blocks may be padded in excess of their dimension. Any
        // additional blocks will need to have their stride adjusted.
        dim_t outer_stride = 1;
        for (const auto &eb : l.enumerated_blocks()) {
            const auto &b = eb.second;
            const auto stride = std::max((dim_t)b.stride, outer_stride);
            auto &dim = dims_[b.dim_idx];
            auto &dim_state = state[b.dim_idx];
            if (!dim_state.has_next()) continue;
            auto block = *state[b.dim_idx];
            if (l.is_outermost(eb) && !strict) {
                outer_stride = stride * dim;
                if (l.ndims() > 1 && dim >= 16) continue;
                options.emplace_back(b.dim_idx, block, outer_stride, l.type());
                break;
            }
            if (skip_mask & (1 << b.dim_idx)) continue;
            if (block > max_block_elems) {
                skip_mask |= (1 << b.dim_idx);
                continue;
            }
            if (dim % b.block == 0) {
                dim /= b.block;
                continue;
            }
            auto step = math::gcd(dim, b.block);
            options.emplace_back(b.dim_idx, block, stride * step, l.type());
            break;
        }
    }

    auto cmp = [&](const option_t &l, const option_t &r) {
        return l.type.size() * l.block * l.stride
                < r.type.size() * r.block * r.stride;
    };
    std::sort(options.begin(), options.end(), cmp);

    for (auto &o : options) {
        if (o.block <= max_block_elems) return o;
    }
    return {(dim_idx_t)-1, 0, 0};
}

dim_t tile_size(
        const std::vector<layout_t> &layouts, const std::vector<dim_t> &dims) {
    auto layout_tile_size = [&](const layout_t &layout) {
        dim_t elems = 1;
        auto rem_dims = dims;
        for (auto &eb : layout.enumerated_blocks()) {
            auto &block = eb.second;
            auto &rem_dim = rem_dims[block.dim_idx];
            if (layout.is_outermost(eb) || block.block % rem_dim == 0) {
                elems *= utils::rnd_up_pow2(rem_dim);
                rem_dim = 1;
            } else if (rem_dim % block.block == 0) {
                elems *= utils::rnd_up_pow2(block.block);
                rem_dim /= block.block;
            } else {
                gpu_assert(!"unexpected tile");
            }
        }
        return elems;
    };

    dim_t elems = 0;
    for (auto &l : layouts)
        elems = std::max(elems, layout_tile_size(l));
    return elems;
}

std::vector<tensor_t> tiles(const std::vector<layout_t> &layouts,
        const std::vector<uint32_t> &masks, dim_t max_elems, bool strict) {
    if (layouts.empty()) return {tensor_t()};
    dim_idx_t ndims = layouts[0].ndims();
    for (auto &l : layouts)
        if (l.ndims() != ndims) return {tensor_t()};
    if (ndims == 0) return {tensor_t()};

    std::vector<tiling_dim_state_t> state(ndims);
    for (dim_idx_t i = 0; i < ndims; ++i)
        state[i] = {checkpoints(layouts, masks, i, strict)};

    std::vector<dim_t> dims(ndims, 1);
    std::vector<tensor_t> tiles = {{dims}};
    while (true) {
        dim_t elems = tile_size(layouts, dims);
        auto b = next_block(layouts, dims, max_elems / elems, state, strict);
        if (b.dim_idx >= ndims) break;
        ++state[b.dim_idx];
        if (b.block == 1) continue;
        dims[b.dim_idx] *= b.block;
        tiles.emplace_back(dims);
    }
    return tiles;
}

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
