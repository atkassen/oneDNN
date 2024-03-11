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

#ifndef GPU_INTEL_JIT_REORDER_NORMALIZATION_HPP
#define GPU_INTEL_JIT_REORDER_NORMALIZATION_HPP

#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace reorder {

void normalize(std::vector<layout_t> &layouts, std::vector<uint32_t> &masks,
        bool maintain_blocks = false);

inline void normalize(
        std::vector<layout_t> &layouts, bool maintain_blocks = false) {
    std::vector<uint32_t> masks(layouts.size(), -1);
    normalize(layouts, masks, maintain_blocks);
}

inline void normalize(layout_t &a, layout_t &b) {
    std::vector<layout_t> layouts = {a, b};
    normalize(layouts);
    a = layouts[0];
    b = layouts[1];
};

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
