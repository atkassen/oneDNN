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

#ifndef GPU_INTEL_JIT_CODEGEN_CONVERSION_HPP
#define GPU_INTEL_JIT_CODEGEN_CONVERSION_HPP

#include "common/utils.hpp"
#include "gpu/intel/jit/codegen/operand.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/ngen/ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class conversion_mark_t { unvisited, visited };

struct conversion_stage_t;

struct conversion_chunk_t {
    int offset;
    int stride;
    int width;
    ngen::DataType type;

    std::shared_ptr<conversion_stage_t> next = nullptr;
    conversion_mark_t mark = conversion_mark_t::unvisited;
};

struct conversion_stage_t {
    std::vector<conversion_chunk_t> chunks;

    void set_next(const std::shared_ptr<conversion_stage_t> &p);
};

void visit(std::shared_ptr<conversion_stage_t> &p,
        std::function<void(std::shared_ptr<conversion_stage_t> &p)> f);

struct conversion_plan_t {
    std::shared_ptr<conversion_stage_t> head;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
