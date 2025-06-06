/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"

#include "xpu/ocl/engine_factory.hpp"
#include "xpu/ocl/engine_impl.hpp"
#include "xpu/ocl/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::xpu::ocl;

status_t dnnl_ocl_interop_engine_create(dnnl::impl::engine_t **engine,
        cl_device_id device, cl_context context) {
    bool args_ok = !utils::any_null(engine, device, context);
    VERROR_ENGINE(args_ok, status::invalid_arguments, VERBOSE_NULL_ARG);

    xpu::ocl::engine_factory_t f(engine_kind::gpu);

    size_t index;
    CHECK(xpu::ocl::get_device_index(&index, device));

    return f.engine_create(engine, device, context, index);
}

status_t dnnl_ocl_interop_engine_get_context(
        engine_t *engine, cl_context *context) {
    bool args_ok = !utils::any_null(engine, context)
            && (engine->runtime_kind() == runtime_kind::ocl);

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_engine_impl
            = utils::downcast<const xpu::ocl::engine_impl_t *>(engine->impl());
    *context = ocl_engine_impl->context();
    return status::success;
}

status_t dnnl_ocl_interop_get_device(engine_t *engine, cl_device_id *device) {
    bool args_ok = !utils::any_null(engine, device)
            && (engine->runtime_kind() == runtime_kind::ocl);

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_engine_impl
            = utils::downcast<const xpu::ocl::engine_impl_t *>(engine->impl());
    *device = ocl_engine_impl->device();
    return status::success;
}

status_t dnnl_ocl_interop_engine_create_from_cache_blob(engine_t **engine,
        cl_device_id device, cl_context context, size_t size,
        const uint8_t *cache_blob) {
    bool args_ok = !utils::any_null(engine, device, context, cache_blob)
            && size != 0;
    VERROR_ENGINE(args_ok, status::invalid_arguments, VERBOSE_NULL_ARG);

    xpu::ocl::engine_factory_t f(engine_kind::gpu);

    size_t index;
    CHECK(xpu::ocl::get_device_index(&index, device));

    const std::vector<uint8_t> cb(cache_blob, cache_blob + size);
    return f.engine_create(engine, device, context, index, cb);
}

status_t dnnl_ocl_interop_engine_get_cache_blob(
        engine_t *engine, size_t *size, uint8_t *cache_blob) {
    if (engine->kind() != engine_kind::gpu || !size)
        return status::invalid_arguments;

    if (!cache_blob) {
        size_t sz = 0;
        CHECK(engine->get_cache_blob_size(&sz));
        (*size) = sz;
        return status::success;
    }

    CHECK(engine->get_cache_blob(*size, cache_blob));
    return status::success;
}

status_t dnnl_ocl_interop_engine_get_cache_blob_id(
        cl_device_id device, size_t *size, uint8_t *cache_blob) {
    if (size == nullptr) return status::invalid_arguments;
    size_t &id_size = *size;

    serialization_stream_t sstream;

    size_t platform_name_size = 0;
    size_t device_name_size = 0;
    size_t driver_version_size = 0;

    cl_int err = CL_SUCCESS;

    // Get oneDNN version.
    auto version = dnnl_version();

    // Get platform.
    cl_platform_id platform;
    err = clGetDeviceInfo(
            device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    OCL_CHECK(err);

    // Get platform name size.
    err = clGetPlatformInfo(
            platform, CL_PLATFORM_NAME, 0, nullptr, &platform_name_size);
    OCL_CHECK(err);

    // Get device name size.
    err = clGetDeviceInfo(
            device, CL_DEVICE_NAME, 0, nullptr, &device_name_size);
    OCL_CHECK(err);

    // Get driver version size.
    err = clGetDeviceInfo(
            device, CL_DRIVER_VERSION, 0, nullptr, &driver_version_size);
    OCL_CHECK(err);

    if (!cache_blob) {
        id_size = platform_name_size + device_name_size + driver_version_size
                + sizeof(version->major) + sizeof(version->minor)
                + sizeof(version->patch) + std::strlen(version->hash)
                + 4 * sizeof(size_t);
        return status::success;
    }

    // Get platform name.
    auto platform_name = std::string(platform_name_size, '\0');
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name.size(),
            &platform_name[0], nullptr);
    OCL_CHECK(err);

    sstream.append_array(platform_name.size(), platform_name.data());

    // Get device name.
    auto device_name = std::string(device_name_size, '\0');
    err = clGetDeviceInfo(
            device, CL_DEVICE_NAME, device_name_size, &device_name[0], nullptr);
    OCL_CHECK(err);

    sstream.append_array(device_name.size(), device_name.data());

    // Get driver version.
    auto driver_version = std::string(driver_version_size, '\0');
    err = clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_size,
            &driver_version[0], nullptr);
    OCL_CHECK(err);

    sstream.append_array(driver_version.size(), driver_version.data());

    // Get oneDNN version.
    sstream.append(version->major);
    sstream.append(version->minor);
    sstream.append(version->patch);

    // Get oneDNN hash.
    sstream.append_array(std::strlen(version->hash), version->hash);

    // Not enough buffer space for copying cache blob.
    if (id_size != sstream.get_data().size()) return status::invalid_arguments;

    std::memcpy(cache_blob, sstream.get_data().data(), id_size);

    return status::success;
}
