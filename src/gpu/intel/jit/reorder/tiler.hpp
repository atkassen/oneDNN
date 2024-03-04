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

#ifndef DNNL_GPU_INTEL_JIT_REORDER_TILER_HPP
#define DNNL_GPU_INTEL_JIT_REORDER_TILER_HPP

#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

std::vector<tensor_t> tiles(const std::vector<layout_t> &layouts,
        const std::vector<uint32_t> &masks, dim_t max_elems = 256,
        bool strict = false);

inline tensor_t tile(const std::vector<layout_t> &layouts,
        const std::vector<uint32_t> &masks, dim_t max_elems = 256,
        bool strict = false) {
    return tiles(layouts, masks, max_elems, strict).back();
}

inline std::vector<tensor_t> tiles(const layout_t &src, const layout_t &dst,
        dim_t max_elems = 256, bool strict = false) {
    std::vector<uint32_t> masks(2, -1);
    return tiles({src, dst}, masks, max_elems, strict);
}

inline tensor_t tile(const layout_t &src, const layout_t &dst,
        dim_t max_elems = 256, bool strict = false) {
    return tiles(src, dst, max_elems, strict).back();
}

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
