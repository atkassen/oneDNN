/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "xpu/usm_utils.hpp"
#include "common/z_magic.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace usm {

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define DNNL_WITH_ASAN
#endif
#endif

void poison(void *ptr, size_t size) {
#ifdef DNNL_WITH_ASAN
    if (ptr == nullptr || size == 0) return;
    __asan_poison_memory_region(ptr, size);
#endif
    MAYBE_UNUSED(ptr);
    MAYBE_UNUSED(size);
}

void unpoison(void *ptr, size_t size) {
#ifdef DNNL_WITH_ASAN
    if (ptr == nullptr || size == 0) return;
    __asan_unpoison_memory_region(ptr, size);
#endif
    MAYBE_UNUSED(ptr);
    MAYBE_UNUSED(size);
}

#ifdef DNNL_WITH_ASAN
#undef DNNL_WITH_ASAN
#endif

} // namespace usm
} // namespace xpu
} // namespace impl
} // namespace dnnl
