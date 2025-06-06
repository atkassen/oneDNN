GELUBackward {#dev_guide_op_gelubackward}
=========================================

## General

GELUBackward operation computes gradient for GELU.

## Operation attributes

| Attribute Name                             | Description                                     | Value Type | Supported Values                  | Required or Optional |
|:-------------------------------------------|:------------------------------------------------|:-----------|:----------------------------------|:---------------------|
| [mode](@ref dnnl::graph::op::attr::mode)   | Specifies the computation mode of GELUBackward. | string     | `gelu_erf` (default), `gelu_tanh` | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

GELUBackward operation supports the following data type combinations.

| Src  | Diff_dst | Diff_src |
|:-----|:---------|:---------|
| f32  | f32      | f32      |
| f16  | f16      | f16      |
| bf16 | bf16     | bf16     |
