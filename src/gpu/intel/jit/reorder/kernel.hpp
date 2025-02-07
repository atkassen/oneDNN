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

#ifndef GPU_INTEL_JIT_REORDER_KERNEL_HPP
#define GPU_INTEL_JIT_REORDER_KERNEL_HPP

#include "gpu/intel/jit/codegen/codegen.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/codegen/ngen_helpers.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/ir/ir_builder.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/reorder/ir_builder.hpp"
#include "gpu/intel/jit/reorder/kernel_desc.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

namespace reorder {

template <ngen::HW hw = ngen::HW::Unknown>
class kernel_t : public ir_kernel_t<hw> {
public:
    IR_KERNEL_FORWARD(hw)

    kernel_t(const kernel_desc_base_t &desc, const kernel_info_t &kernel_info)
        : ir_kernel_t<hw>(desc, kernel_info) {
        //auto &rdesc = static_cast<const kernel_desc_t &>(desc);

        this->require_signal_header_ = true;

        // Build IR for the kernel.
        grid_context_t grid_ctx;
        stmt_t body; // = build_ir(rdesc, kernel_info, grid_ctx);

        alloc_manager_t alloc_mgr(body);
        setup_interface(body);

        generate_prologue();

        // Bind "external" variables
        expr_binding_t expr_binding(hw);
        bind_external_vars(body, grid_ctx, expr_binding);

        // Generate assembly from IR.
        convert_ir_to_ngen<hw>(body, this, expr_binding);

        generate_epilogue();
    }
};

} // namespace reorder
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
