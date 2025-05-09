CPU Dispatcher Control {#dev_guide_cpu_dispatcher_control}
==========================================================

oneDNN uses JIT code generation to implement most of its functionality and will
choose the best code based on detected processor features. Sometimes it is
necessary to control which features oneDNN detects. This is sometimes useful for
debugging purposes or for performance exploration. For example, test SSE4.1 code
on an AVX2-capable processor.

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`ONEDNN_ENABLE_MAX_CPU_ISA`.

| CMake Option              | Supported values (defaults in bold) | Description                                                              |
|:--------------------------|:------------------------------------|:-------------------------------------------------------------------------|
| ONEDNN_ENABLE_MAX_CPU_ISA | **ON**, OFF                         | Enables [CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control) |

## Runtime Controls

When the feature is enabled at build-time, the `ONEDNN_MAX_CPU_ISA` environment
variable can be used to limit processor features oneDNN is able to detect to
certain Instruction Set Architecture (ISA) and older instruction sets. It can
also be used to enable ISAs with initial support in the library that are
otherwise disabled by default. When the feature is disabled at build-time, the
environment variable doesn't take any effect but: 1) the detection will still
happen following the default behavior; 2) functions to control the behavior
still take effect.

| Environment variable | Value                                        | Description                                                                                                                                                           |
|:---------------------|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ONEDNN_MAX_CPU_ISA   | SSE41                                        | Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)                                                                                                                    |
| \                    | AVX                                          | Intel Advanced Vector Extensions (Intel AVX)                                                                                                                          |
| \                    | AVX2                                         | Intel Advanced Vector Extensions 2 (Intel AVX2)                                                                                                                       |
| \                    | AVX2_VNNI                                    | Intel AVX2 with Intel Deep Learning Boost (Intel DL Boost)                                                                                                            |
| \                    | AVX512_CORE                                  | Intel AVX-512 with AVX512BW, AVX512VL, and AVX512DQ extensions                                                                                                        |
| \                    | AVX512_CORE_VNNI                             | Intel AVX-512 with Intel DL Boost                                                                                                                                     |
| \                    | AVX512_CORE_BF16                             | Intel AVX-512 with Intel DL Boost and bfloat16 support                                                                                                                |
| \                    | AVX10_1_512 or AVX512_CORE_FP16              | Intel AVX10.1/512 with float16 and Intel DL Boost and bfloat16                                                                                                        |
| \                    | AVX10_1_512_AMX or AVX512_CORE_AMX           | Intel AVX10.1/512 with float16, Intel DL Boost and bfloat16 support and Intel Advanced Matrix Extensions (Intel AMX) with 8-bit integer and bfloat16 support          |
| \                    | AVX2_VNNI_2                                  | Intel AVX2 with Intel DL Boost with 8-bit integer, float16 and bfloat16 support                                                                                       |
| \                    | AVX10_1_512_AMX_FP16 or AVX512_CORE_AMX_FP16 | Intel AVX10.1/512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer, bfloat16 and float16 support                                    |
| \                    | **DEFAULT**                                  | **No restrictions on the above ISAs, but excludes the below ISAs with preview support in the library (default)**                                                      |
| \                    | AVX10_2_512                                  | Intel AVX10.2/512 with float16, Intel DL Boost and bfloat16                                                                                                           |
| \                    | AVX10_2_512_AMX_2                            | Intel AVX10.2/512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with 8-bit integer, bfloat16, float16, float8 support                               |

@note The ISAs are partially ordered:
* SSE41 < AVX < AVX2 < AVX2_VNNI < AVX2_VNNI_2,
* AVX2 < AVX512_CORE < AVX512_CORE_VNNI < AVX512_CORE_BF16
  < AVX10_1_512 < AVX10_2_512,
* AVX10_1_512 < AVX10_1_512_AMX < AVX10_1_512_AMX_FP16
  < AVX10_2_512_AMX_2,
* AVX2_VNNI < AVX10_1_512,
* AVX10_2_512 < AVX10_2_512_AMX_2.

The below values are aliased for backward compatibility with the introduction of
AVX10.1:
* AVX512_CORE_FP16 = AVX10_1_512
* AVX512_CORE_AMX = AVX10_1_512_AMX
* AVX512_CORE_AMX_FP16 = AVX10_1_512_AMX_FP16

This feature can also be managed at runtime with the following functions:
* @ref dnnl::set_max_cpu_isa function allows changing the ISA at runtime. The
  limitation is that it is possible to set the value only once. This ensures
  that the JIT-ed code observe consistent CPU features both during generation
  and execution. In addition, it is advised to call this function before any
  other oneDNN API. This is because the first internal ISA query will disable
  the ability to change the ISA. Once disabled, changing the ISA will return an
  error.
* @ref dnnl::get_effective_cpu_isa function returns the currently used CPU ISA
  which is the highest available CPU ISA by default.

Function settings take precedence over environment variables.
