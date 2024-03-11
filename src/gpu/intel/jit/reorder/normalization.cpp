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

#include "gpu/intel/jit/reorder/normalization.hpp"
#include "gpu/intel/compute/block_manipulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

namespace merge {
struct type_t {
    dim_t block;
    bool complete;

    constexpr bool operator==(const type_t &other) const {
        return block == other.block && complete == other.complete;
    }
};

constexpr type_t undef {0, true};
constexpr type_t none {1, true};
} // namespace merge

struct tile_info_t {
    size_t idx;
    block_t outer, inner;
    std::array<dim_t, 2> tile;
    merge::type_t merge;

    tile_info_t(size_t idx, block_t outer, block_t inner,
            std::array<dim_t, 2> tile, merge::type_t merge = merge::undef)
        : idx(idx), outer(outer), inner(inner), tile(tile), merge(merge) {}

    bool is_dense() const { return outer.stride == inner.stride * inner.block; }
};

std::ostream &operator<<(std::ostream &out, const tile_info_t &info) {
    return out << info.idx << ' ' << info.outer.str() << ' ' << info.inner.str()
               << ' ' << info.tile[0] << 'x' << info.tile[1]
               << " block: " << info.merge.block;
}

inline bool mask_set(uint32_t bits, uint32_t mask) {
    return (bits & mask) == mask;
}

inline bool bit_set(uint32_t bits, dim_idx_t index) {
    gpu_assert(index < 32);
    return mask_set(bits, 1 << index);
}

void compress(std::vector<uint32_t> &bits, uint32_t mask) {
    // from Hacker's Delight 7-4
    uint32_t mk = ~mask << 1, mp, mv[5], t;

    for (auto &b : bits)
        b &= mask;

    for (int i = 0; i < 5; ++i) {
        mp = mk ^ (mk << 1);
        mp = mp ^ (mp << 2);
        mp = mp ^ (mp << 4);
        mp = mp ^ (mp << 8);
        mp = mp ^ (mp << 16);
        mv[i] = mp & mask;
        mask = (mask ^ mv[i]) | (mv[i] >> (1 << i));
        mk = mk & ~mp;
    }

    for (auto &b : bits) {
        for (int i = 0; i < 5; ++i) {
            t = b & mv[i];
            b = (b ^ t) | (t >> (1 << i));
        }
    }
}

void compress(uint32_t &bits, uint32_t mask) {
    std::vector<uint32_t> bits_vec = {bits};
    compress(bits_vec, mask);
    bits = bits_vec[0];
}

merge::type_t get_merge(const std::vector<tile_info_t> &infos,
        uint32_t broadcast_mask, uint32_t merge_mask) {
    if (infos.empty()) return merge::none;

    auto merge_ok = [&](dim_idx_t i, dim_idx_t j) {
        return !(bit_set(broadcast_mask, i) ^ bit_set(broadcast_mask, j))
                && bit_set(merge_mask, i) && bit_set(merge_mask, j);
    };
    const auto &ref = infos[0];
    if (!merge_ok(ref.inner.dim_idx, ref.outer.dim_idx)) return merge::none;
    auto get_info_merge = [&](const tile_info_t &info) {
        if (!info.is_dense()) return merge::none;
        if (info.inner.dim_idx != ref.inner.dim_idx) return merge::none;
        if (info.outer.dim_idx != ref.outer.dim_idx) return merge::none;
        if (info.tile[0] != ref.tile[0]) return merge::none;
        if (info.tile[1] != ref.tile[1]) return merge::none;
        auto block = math::gcd(info.outer.block, ref.outer.block);
        bool complete
                = utils::everyone_is(block, info.outer.block, ref.outer.block);
        return merge::type_t {block, complete};
    };

    merge::type_t merge = merge::undef;
    for (size_t i = 1; i < infos.size(); ++i) {
        auto info_merge = get_info_merge(infos[i]);
        merge = {math::gcd(merge.block, info_merge.block),
                merge.complete && info_merge.complete};
        if (merge == merge::none) return merge::none;
    }
    return merge;
}

layout_t reduce(const layout_t &src) {
    std::vector<block_t> blocks;
    for (auto &b : src.blocks()) {
        if (b.block != 1) {
            if (blocks.empty() || blocks.back().dim_idx != b.dim_idx) {
                blocks.push_back(b);
            } else {
                blocks.back().block *= b.block;
            }
        }
    }
    if (blocks.empty()) blocks.emplace_back(0, 1, 1);
    return {src.type(), src.ndims(), src.offset(), blocks,
            /*do_normalize=*/false};
}

std::vector<tile_info_t> tile_info(
        const layout_t &layout, uint32_t &blocking_mask) {
    using tile_t = std::array<dim_t, 2>;
    if (layout.blocks().empty()) return {};

    auto cmp = [](const tile_info_t &l, const tile_info_t &r) {
        return l.outer.dim_idx < r.outer.dim_idx;
    };

    uint32_t seen_dims = 0;
    std::vector<dim_t> tile(layout.ndims(), 1);
    std::vector<tile_info_t> infos;
    infos.reserve(layout.blocks().size() - 1);
    auto inner = layout.blocks()[0];
    seen_dims |= (1 << inner.dim_idx);
    for (size_t i = 1; i < layout.blocks().size(); ++i) {
        const auto &outer = layout.blocks()[i];
        uint32_t outer_dim_mask = 1 << outer.dim_idx;
        blocking_mask |= seen_dims & outer_dim_mask;
        seen_dims |= outer_dim_mask;
        tile[inner.dim_idx] *= inner.block;

        tile_t info_tile = {tile[outer.dim_idx], tile[inner.dim_idx]};
        infos.emplace_back(i - 1, outer, inner, info_tile);
        inner = outer;
    }
    std::stable_sort(infos.begin(), infos.end(), cmp);
    return infos;
}

// Compute blocks resulting from performing merges
std::vector<block_t> combine(std::vector<tile_info_t> info) {
    std::vector<block_t> blocks;
    if (info.empty()) return blocks;

    auto cmp = [](const tile_info_t &l, const tile_info_t &r) {
        return l.idx < r.idx;
    };
    std::sort(info.begin(), info.end(), cmp);

    blocks.push_back(info[0].inner);
    for (auto &i : info) {
        if (i.merge.block > merge::none.block) {
            auto &last = blocks.back();
            last.block *= i.merge.block;
            if (i.merge.complete) continue;
            i.outer.block /= i.merge.block;
            i.outer.stride *= i.merge.block;
        }
        blocks.push_back(i.outer);
    }

    return blocks;
}

// Find pairs of consecutive blocks which can be combined
void find_merges(std::vector<std::vector<tile_info_t>> &infos,
        const std::vector<uint32_t> &masks, uint32_t merge_mask) {
    if (infos.empty()) return;

    dim_idx_t ndims = 0;
    using iterator_t = typename std::vector<tile_info_t>::iterator;
    std::vector<iterator_t> it;
    it.reserve(infos.size());
    for (auto &info : infos) {
        it.push_back(info.begin());
        if (info.empty()) continue;
        ndims = std::max(info.back().outer.dim_idx + 1, ndims);
    }

    uint32_t broadcast_mask = -1;
    for (auto &mask : masks)
        broadcast_mask &= mask;

    std::vector<tile_info_t> tiles;
    tiles.reserve(infos.size());
    for (dim_idx_t i = 0; i < ndims; ++i) {
        auto at_end = [=](const iterator_t &it, const iterator_t &end) {
            return it == end || it->outer.dim_idx != i;
        };

        std::vector<iterator_t> end = it;
        for (size_t j = 0; j < infos.size(); ++j) {
            iterator_t &iter = end[j];
            while (!at_end(iter, infos[j].end()))
                ++iter;
        }

        auto get_active_idxs = [&]() {
            using indices_t = std::vector<size_t>;
            indices_t active_idxs;
            for (size_t j = 0; j < infos.size(); ++j) {
                if (!bit_set(masks[j], i)) continue;
                active_idxs.push_back(j);
            }
            return active_idxs;
        };

        auto more_blocks = [&](const std::vector<size_t> &idxs) {
            if (idxs.empty()) return false;
            for (auto j : idxs)
                if (it[j] == end[j]) return false;
            return true;
        };

        auto active_idxs = get_active_idxs();
        while (more_blocks(active_idxs)) {
            tiles.clear();
            for (auto j : active_idxs)
                tiles.push_back(*it[j]);
            auto merge = get_merge(tiles, broadcast_mask, merge_mask);
            for (auto j : active_idxs)
                (it[j]++)->merge = merge;
        }

        // Advance all iterators to the end, which are the beginnings for the
        // next dimension.
        std::swap(end, it);
    }
}

// Find dimensions present in any normalized layout and construct map of new
// dimension indices
void relabel(std::vector<layout_t> &layouts, std::vector<uint32_t> &masks) {
    uint32_t dim_mask = 0;
    std::vector<uint32_t> missing_dim_masks;
    missing_dim_masks.reserve(layouts.size());
    for (auto &l : layouts) {
        uint32_t layout_dim_mask = 0;
        for (auto &b : l.blocks())
            layout_dim_mask |= (1 << b.dim_idx);
        missing_dim_masks.push_back(layout_dim_mask);
        dim_mask |= layout_dim_mask;
    }
    compress(masks, dim_mask);

    // Force at least one dimension
    if (dim_mask == 0) dim_mask = 1;

    for (auto &m : missing_dim_masks)
        m = dim_mask & ~m;

    std::vector<int> dim_map;
    dim_idx_t ndims = 0;
    while (dim_mask) {
        dim_map.push_back(ndims);
        ndims += dim_mask & 1;
        dim_mask >>= 1;
    }

    for (size_t i = 0; i < layouts.size(); ++i) {
        auto &l = layouts[i];
        auto missing_dim_mask = missing_dim_masks[i];
        std::vector<block_t> blocks;
        blocks.reserve(l.blocks().size());
        stride_t stride = 1;
        for (auto b : l.blocks()) {
            b.dim_idx = dim_map[b.dim_idx];
            blocks.push_back(b);
            stride = b.stride * b.block;
        }

        // Add missing dimensions as outer size-1 blocks in reverse lex order
        size_t insertion_index = blocks.size();
        while (missing_dim_mask) {
            uint32_t dim_bit = missing_dim_mask & ~(missing_dim_mask - 1);
            dim_idx_t dim_idx = dim_map[math::ilog2q(dim_bit)];
            block_t b {dim_idx, 1, stride};
            blocks.insert(blocks.begin() + insertion_index, b);
            missing_dim_mask &= ~dim_bit;
        }

        l = {l.type(), ndims, l.offset(), blocks, /*do_normalize=*/false};
    }
}

// Given a vector of layouts, finds an equivalent vector of simpler layouts by
// attempting to combine consecutive blocks that appear in all layouts at the
// same level of nesting for the dimensions to which the blocks belong. E.g.,
//
//            1.           2.
// 16a16b16c ---> 256a16c ---> 256a16b
// 16c16a16b ---> 16c256a ---> 16b256a
//
// 1. The consecutive blocks 16a16b are repeated. For the first layout it
//    appears with an inner tile 1x1x16, and 1x1x1 for the second. Because the
//    AB-subtile is 1x1 for both and the inner block (16b) is the same for
//    both, we can combine these blocks (in merge::type_t parlance, this is a
//    forward merge). In other cases, where including the inner block into the
//    tile creates equal subtiles, we can also combine blocks (backward merge).
// 2. The B dimension no longer appears, so we can remove it from the layout and
//    re-label the dimensions so that the new layouts are 2D.
void normalize(std::vector<layout_t> &layouts,
        std::vector<uint32_t> &broadcast_masks, bool maintain_blocks) {
    std::vector<std::vector<tile_info_t>> infos;
    gpu_assert(layouts.size() == broadcast_masks.size());
    size_t ntensors = layouts.size();
    uint32_t blocking_mask = 0;
    infos.reserve(ntensors);
    for (size_t i = 0; i < ntensors; ++i)
        infos.push_back(tile_info(reduce(layouts[i]), blocking_mask));
    if (!maintain_blocks) blocking_mask = 0;
    find_merges(infos, broadcast_masks, ~blocking_mask);

    for (size_t i = 0; i < layouts.size(); ++i) {
        const auto &info = infos[i];
        auto &layout = layouts[i];
        auto blocks = combine(info);
        if (!info.empty())
            layout = {layout.type(), layout.ndims(), layout.offset(), blocks,
                    /*do_normalize=*/false};
    }

    relabel(layouts, broadcast_masks);
}

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
