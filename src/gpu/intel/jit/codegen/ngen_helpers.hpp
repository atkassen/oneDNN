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

#ifndef GPU_INTEL_JIT_CODEGEN_NGEN_HELPERS_HPP
#define GPU_INTEL_JIT_CODEGEN_NGEN_HELPERS_HPP

#include "gpu/intel/jit/ir/core.hpp"
#include "ngen.hpp"
#include "ngen_register_allocator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

constexpr ngen::DataType ngen_f4_e3m0() {
    return static_cast<ngen::DataType>(0x5B);
}

constexpr ngen::DataType ngen_f4_e2m1() {
    return static_cast<ngen::DataType>(0x5A);
}

template <typename T>
T to_cpp(const ngen::Immediate &imm) {
    auto u64 = uint64_t(imm);
    switch (imm.getType()) {
        case ngen::DataType::w:
            return (T)utils::bit_cast<std::array<int16_t, 4>>(u64)[0];
        case ngen::DataType::uw:
            return (T)utils::bit_cast<std::array<uint16_t, 4>>(u64)[0];
        case ngen::DataType::d:
            return (T)utils::bit_cast<std::array<int32_t, 2>>(u64)[0];
        case ngen::DataType::ud:
            return (T)utils::bit_cast<std::array<uint32_t, 2>>(u64)[0];
        case ngen::DataType::q: return (T)utils::bit_cast<int64_t>(u64);
        case ngen::DataType::uq: return (T)utils::bit_cast<uint64_t>(u64);
        case ngen::DataType::f:
            return (T)utils::bit_cast<std::array<float, 2>>(u64)[0];
        default: gpu_error_not_expected();
    }
    return 0;
}

// type_t to ngen::DataType convertor.
inline ngen::DataType to_ngen(const type_t &type) {
    gpu_assert(type.is_scalar()) << "Expected scalar type.";

#define CASE(_kind, ngen_enum) \
    if (type.kind() == type_kind_t::_kind) return ngen::DataType::ngen_enum

    // Until f4_e3m0 lands in ngen
    if (type.kind() == type_kind_t::f4_e3m0) return ngen_f4_e3m0();
    // Until f4_e2m1 lands in ngen
    if (type.kind() == type_kind_t::f4_e2m1) return ngen_f4_e2m1();

    CASE(bf16, bf);
    CASE(f16, hf);
    CASE(bf8, bf8);
    CASE(hf8, hf8);
    CASE(tf32, tf32);
    CASE(f32, f);
    CASE(f64, df);
    CASE(s16, w);
    CASE(s32, d);
    CASE(s64, q);
    CASE(s8, b);
    CASE(s4, s4);
    CASE(u16, uw);
    CASE(u32, ud);
    CASE(u64, uq);
    CASE(u8, ub);
    CASE(u4, u4);

    if (type == type_t::byte_ptr()) return ngen::DataType::uq;

#undef CASE
    gpu_error_not_expected();
    return ngen::DataType::invalid;
}

// ngen::DataType to type_t convertor.
inline type_t to_ir(ngen::DataType type) {
#define CASE(_kind, ngen_enum) \
    if (type == ngen::DataType::ngen_enum) return type_t::_kind();

    CASE(bf16, bf);
    CASE(f16, hf);
    CASE(bf8, bf8);
    CASE(hf8, hf8);
    CASE(f32, f);
    CASE(f64, df);
    CASE(s16, w);
    CASE(s32, d);
    CASE(s64, q);
    CASE(s8, b);
    CASE(s4, s4);
    CASE(u16, uw);
    CASE(u32, ud);
    CASE(u64, uq);
    CASE(u8, ub);
    CASE(u4, u4);

#undef CASE
    gpu_error_not_expected();
    return type_t::undef();
}

inline ngen::Immediate to_ngen(
        const expr_t &expr, const type_t &type = type_t::undef()) {
    gpu_assert(expr.type().is_scalar()) << "Vector types are not supported.";
    if (expr.is<int_imm_t>()) {
        auto &imm = expr.as<int_imm_t>();
        // No conversion.
        if (utils::one_of(type, type_t::undef(), expr.type()))
            return ngen::Immediate(imm.value);
            // Do conversion.
#define CASE(cpp_type) \
    if (type.is_cpp<cpp_type>()) return ngen::Immediate(cpp_type(imm.value))

        CASE(int16_t);
        CASE(int32_t);
        CASE(int64_t);
        CASE(uint16_t);
        CASE(uint32_t);
        CASE(uint64_t);

#undef CASE
        gpu_error_not_expected() << "Can't convert expression: " << expr;
    } else if (expr.is<float_imm_t>()) {
        gpu_assert(utils::one_of(type, type_t::undef(), type_t::f32()))
                << "Conversion is not supported.";
        auto &imm = expr.as<float_imm_t>();
        if (imm.type.is_f32()) { return ngen::Immediate((float)imm.value); }
        return ngen::Immediate(imm.value);
    }
    gpu_error_not_expected() << "Can't convert expression: " << expr;
    return ngen::Immediate();
}

inline ngen::Immediate ngen_negate(const ngen::Immediate &imm) {
    switch (imm.getType()) {
        case ngen::DataType::w: return ngen::Immediate(-to_cpp<int16_t>(imm));
        case ngen::DataType::d: return ngen::Immediate(-to_cpp<int32_t>(imm));
        case ngen::DataType::f: return ngen::Immediate(-to_cpp<float>(imm));
        default: gpu_error_not_expected();
    }
    return ngen::Immediate();
}

inline bool ngen_is_qw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::q, ngen::DataType::uq);
}

inline bool ngen_is_dw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::d, ngen::DataType::ud);
}

inline bool ngen_is_w(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::w, ngen::DataType::uw);
}

inline bool ngen_is_b(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::b, ngen::DataType::ub);
}

inline bool ngen_is_xf(ngen::DataType type) {
    return utils::one_of(
            type, ngen::DataType::bf, ngen::DataType::hf, ngen::DataType::f);
}

inline ngen::Subregister get_subregister(
        ngen::HW hw, ngen::DataType type, const ngen::GRFRange &r, int idx) {
    int grf_bits = ngen::GRF::bytes(hw) * 8;
    int type_bits = ngen::getBits(type);
    int off_bits = idx * type_bits;
    return r[off_bits / grf_bits].sub((off_bits % grf_bits) / type_bits, type);
}

inline ngen::Subregister get_subregister(const ngen::RegData &rd) {
    return ngen::Subregister(rd, rd.getOffset(), rd.getType());
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
