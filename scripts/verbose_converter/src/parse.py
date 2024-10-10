################################################################################
# Copyright 2024-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import enum
import string
from contextlib import nullcontext
from typing import (
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from . import ir

__all__ = ["Parser"]


class ParseSpec:
    digits = list(string.digits)

    def __init__(self, buf: str):
        self._buf = buf
        self.offset = 0

    def __str__(self):
        return self.buf

    @property
    def buf(self):
        return self._buf[self.offset :]

    @property
    def eof(self):
        return self.offset >= len(self._buf)

    def peek(self, n=1):
        return self.buf[:n]

    def seek(self, n=1):
        self._read(n)

    def _read(self, n: int) -> str:
        token = self._buf[self.offset : self.offset + n]
        self.offset += n
        return token

    def _find_str(self) -> int:
        buf = ParseSpec(self.buf)
        while not buf.eof and buf.peek() not in ("+", ":"):
            buf.seek()
        return buf.offset

    def _find_uint(self) -> int:
        buf = ParseSpec(self.buf)
        if buf.eof or buf.peek() not in self.digits:
            return 0

        if not buf.read_literal("0"):
            while buf.read_one_of(*self.digits):
                pass
        return buf.offset

    def _find_int(self) -> int:
        buf = ParseSpec(self.buf)
        buf.read_one_of("-", "+")
        return buf.offset + buf._find_uint()

    def _find_float(self) -> int:
        buf = ParseSpec(self.buf)
        buf.read_one_of("-", "+")
        if buf.eof or buf.peek() not in ["."] + self.digits:
            return 0  # ignore [+/-][e...]
        if not buf.read_literal("0"):
            while buf.read_one_of(*self.digits):
                pass
        # else: we already read a 0.
        if buf.read_literal("."):
            while buf.read_one_of(*self.digits):
                pass
        if buf.read_literal("e"):
            buf.read_one_of("-", "+")
            if not buf.read_one_of(*self.digits):
                return 0  # ignore [+/-][X][.Y]e[+/-]
            while buf.read_one_of(*self.digits):
                pass
        return buf.offset

    def _find_literal(self, literal):
        if self.buf.startswith(literal):
            return len(literal)
        return 0

    def read_str(self) -> str:
        return self._read(self._find_str())

    def read_literal(self, literal: str) -> Optional[str]:
        offset = self._find_literal(literal)
        if offset == len(literal):
            return self._read(offset)
        return None

    def read_one_of(self, *literals: str) -> Optional[str]:
        for literal in literals:
            if self.read_literal(literal) is not None:
                return literal
        return None

    def read_uint(self) -> Optional[int]:
        offset = self._find_uint()
        if offset:
            return int(self._read(offset))
        return None

    def read_int(self) -> Optional[int]:
        offset = self._find_int()
        if offset:
            return int(self._read(offset))
        return None

    def read_float(self) -> Optional[float]:
        offset = self._find_float()
        if offset:
            return float(self._read(offset))
        return None


class ParseError(ValueError):
    pass


class InvalidEntryError(ParseError):
    pass


class Component(enum.Enum):
    PRIMITIVE = "primitive"
    GRAPH = "graph"


class ParserImpl:
    default_primitive_template = (
        "operation,engine,primitive,implementation,prop_kind,"
        + "memory_descriptors,attributes,auxiliary,problem_desc,exec_time"
    )
    default_graph_template = (
        "operation,engine,partition_id,partition_kind,operations,data_formats,"
        "logical_tensors,fpmath_mode,implementation,backend,exec_time"
    )
    _impl_map: Dict[int, type] = {}

    @staticmethod
    def parse_aux(aux: str):
        parsed: Dict[str, str] = {}
        if aux == "":
            return parsed
        for aux_l in aux.split():
            # Handle strings like NAME:VAL1[:VAL2[:VAL3...]]
            field, *values = aux_l.split(":", 1)
            parsed[field] = values[0] if values else ""
        return parsed

    def parse_mds(self, descriptors):
        try:
            return list(map(self.parse_md, descriptors.split()))
        except ValueError:
            raise ParseError(f"Could not parse mds {descriptors}")

    @staticmethod
    def is_bit_layout(dt):
        buf = ParseSpec(dt)
        if not buf.read_literal("e"):
            return False
        if buf.read_uint() is None:
            return False
        if not buf.read_literal("m"):
            return False
        if buf.read_uint() is None:
            return False
        return buf.eof  # eXmY

    def is_float_type(self, dt):
        buf = ParseSpec(dt)
        buf.read_literal("b")  # ignore b in bf16
        if not buf.read_literal("f"):
            return False
        if buf.read_uint() is None:
            return False
        if buf.eof:
            return True  # bf16, f16, f32, f64
        if not buf.read_literal("_"):
            return False
        return self.is_bit_layout(buf.buf)  # fZ_eXmY

    @staticmethod
    def is_int_type(dt):
        buf = ParseSpec(dt)
        if not buf.read_one_of("u", "s"):
            return False
        if buf.read_uint() is None:
            return False
        return buf.eof

    def is_data_type(self, dt):
        return (
            dt == "undef"
            or self.is_int_type(dt)
            or self.is_float_type(dt)
            or self.is_bit_layout(dt)
        )

    @staticmethod
    def parse_md_flags(flags, fields):
        flags = ir.MemoryDescriptor.Flags(value=flags or "f0")
        for field in fields:
            if field[:3] == "s8m":
                flags.s8_comp_mask = field[3:]
            elif field[:3] == "zpm":
                flags.zp_comp_mask = field[3:]
            elif field[:2] == "sa":
                flags.scale_adjust = float(field[2:])
        return flags

    def parse_md(self, descriptor):
        fields = descriptor.split(":")
        arg_dt, properties, format_kind, tag = fields[:4]
        arg_dt_parts = arg_dt.split("_")
        for i in range(1, len(arg_dt_parts)):
            arg = "_".join(arg_dt_parts[:i])
            dt = "_".join(arg_dt_parts[i:])
            if self.is_data_type(dt):
                break
        else:
            if len(arg_dt_parts) != 1 or not self.is_data_type(arg_dt):
                raise ParseError(
                    f"Could not parse memory descriptor {descriptor}"
                )
            arg, dt = "data", arg_dt

        strides = ""
        if "f" not in fields[4] and format_kind != "undef":
            strides = fields[4]
            flags = self.parse_md_flags(fields[5], fields[6:])
        else:
            flags = self.parse_md_flags(fields[4], fields[5:])
        return ir.MemoryDescriptor(
            arg=arg,
            data_type=dt,
            properties=properties,
            format_kind=format_kind,
            tag=tag,
            strides=strides,
            flags=flags,
        )

    def parse_primitive_attrs(self, attrs):
        exts = ir.PrimitiveAttributes()
        for attr in attrs.split():
            spec = ParseSpec(attr)
            name, args = spec.read_str(), ""
            if spec.read_literal(":"):
                args = spec.buf
            if name in ("attr-acc-mode", "attr-acc"):
                exts.acc_mode = self.parse_acc_mode(args)
            elif name == "attr-deterministic":
                exts.deterministic = self.parse_deterministic(args)
            elif name == "attr-dropout":
                exts.dropout = self.parse_dropout(args)
            elif name == "attr-fpmath":
                exts.fpmath = self.parse_fpmath_mode(args)
            # Kept for compatibility with v2.7 and below.
            elif name == "attr-oscale":
                exts.oscale = self.parse_oscale(args)
            elif name == "attr-post-ops":
                exts.post_ops = self.parse_post_ops(args)
            elif name == "attr-rounding-mode":
                exts.rounding_mode = self.parse_rounding_modes(args)
            elif name == "attr-scales":
                exts.scales = self.parse_scales(args)
            elif name == "attr-scratchpad":
                exts.scratchpad = self.parse_scratchpad_mode(args)
            elif name == "attr-zero-points":
                exts.zero_points = self.parse_zero_points(args)
        return exts

    def parse_post_ops(self, post_ops: str):
        spec = ParseSpec(post_ops)
        parsed: List[ir.PostOp] = []
        while True:
            alg = spec.read_str()
            if alg == "sum":
                parsed.append(self.parse_sum_post_op(spec))
            elif alg == "dw":
                parsed.append(self.parse_dw_post_op(spec))
            elif alg == "prelu":
                parsed.append(self.parse_prelu_post_op(spec))
            elif alg.startswith("eltwise_"):
                parsed.append(self.parse_eltwise_post_op(spec, alg))
            elif alg.startswith("binary_"):
                parsed.append(self.parse_binary_post_op(spec, alg))
            else:
                raise ParseError(f"Unexpected post-op: {alg}")
            if not spec.read_literal("+"):
                break
        return parsed

    @staticmethod
    def parse_sum_post_op(spec) -> ir.SumPostOp:
        post_op = ir.SumPostOp()
        if spec.read_literal(":"):
            post_op.scale = spec.read_float()
        if spec.read_literal(":"):
            post_op.zp = spec.read_int()
        if spec.read_literal(":"):
            post_op.dt = spec.read_str()
        return post_op

    @staticmethod
    def parse_dw_post_op(spec) -> ir.DepthwisePostOp:
        if not spec.read_literal(":"):
            raise ParseError("Expected argument for depthwise post-op")
        ksp = spec.read_str()
        post_op = ir.DepthwisePostOp(ksp=ksp)
        if spec.read_literal(":"):
            post_op.dst_dt = spec.read_str()
        if spec.read_literal(":"):
            post_op.wei_dt = "s8"
            post_op.scales.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.scales.value = spec.read_str()
        return post_op

    @staticmethod
    def parse_prelu_post_op(spec) -> ir.PreLUPostOp:
        post_op = ir.PreLUPostOp()
        if spec.read_literal(":"):
            post_op.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.has_scaleshift = spec.read_str() == "true"
        return post_op

    @staticmethod
    def parse_eltwise_post_op(spec, alg) -> ir.EltwisePostOp:
        post_op = ir.EltwisePostOp(alg=alg)
        if spec.read_literal(":"):
            post_op.alpha = spec.read_float()
        if spec.read_literal(":"):
            post_op.beta = spec.read_float()
        if spec.read_literal(":"):
            post_op.scale = spec.read_float()
        return post_op

    @staticmethod
    def parse_binary_post_op(spec, alg) -> ir.BinaryPostOp:
        if not spec.read_literal(":"):
            raise ParseError("Expected data type for binary post-op")
        dt = spec.read_str()
        post_op = ir.BinaryPostOp(alg=alg, dt=dt)
        if spec.read_literal(":"):
            post_op.mask = spec.read_uint()
        if spec.read_literal(":"):
            post_op.tag = spec.read_str()
        return post_op

    @staticmethod
    def parse_dropout(args: str) -> ir.Dropout:
        return ir.Dropout(tag=args if args else None)

    @staticmethod
    def parse_per_argument(attr, name, parse):
        spec = ParseSpec(attr)
        parsed = {}
        while True:
            arg = spec.read_str()
            if not spec.read_literal(":"):
                raise ParseError(f"Expected mask for {arg} {name}")
            parsed[arg] = parse(spec)
            if not spec.read_literal("+"):
                break
        return parsed

    def parse_scales(self, scales: str):
        return self.parse_per_argument(scales, "scale", self.parse_scale)

    @staticmethod
    def parse_quantization_param(spec, read_value, param_type):
        # Old style: mask[:[value[*]|*]]
        # New style: mask[:data_type[:groups]]
        param = param_type()
        param.mask = spec.read_uint()
        if spec.read_literal(":"):
            value = read_value()
            if value is not None:
                param.value = value
                spec.read_literal("*")
            elif spec.read_literal("*"):
                pass
            elif not spec.eof:  # new style
                param.data_type = spec.read_str()
                if spec.read_literal(":"):
                    param.groups = spec.read_str()
        return param

    # v2.7 and below
    def parse_oscale(self, oscale: str):
        spec = ParseSpec(oscale)
        return self.parse_scale(spec)

    def parse_scale(self, spec) -> ir.Scale:
        return self.parse_quantization_param(spec, spec.read_float, ir.Scale)

    def parse_zero_points(self, zps: str):
        return self.parse_per_argument(zps, "zero point", self.parse_zero_point)

    def parse_zero_point(self, spec) -> ir.ZeroPoint:
        return self.parse_quantization_param(spec, spec.read_int, ir.ZeroPoint)

    @staticmethod
    def parse_fpmath_mode(mathmode: str) -> ir.FPMathMode:
        spec = ParseSpec(mathmode)
        mode = spec.read_str()
        apply_to_int = False
        if spec.read_literal(":"):
            apply_to_int = spec.read_str() == "true"
        return ir.FPMathMode(mode=mode, apply_to_int=apply_to_int)

    @staticmethod
    def parse_rounding_mode(rounding_mode: str) -> ir.RoundingMode:
        rm = rounding_mode.lower()
        for member in ir.RoundingMode.__members__.values():
            if str(member) == rm:
                return member
        else:
            raise ParseError(f"Invalid rounding mode {rounding_mode}")

    def parse_rounding_modes(self, rounding_modes: str):
        spec = ParseSpec(rounding_modes)
        modes: Dict[str, ir.RoundingMode] = {}
        while True:
            arg = spec.read_str()
            if not spec.read_literal(":"):
                raise ParseError("Expected rounding mode")
            mode = self.parse_rounding_mode(spec.read_str())
            modes[arg] = mode
            if not spec.read_literal("+"):
                break
        return modes

    identity = staticmethod(lambda x: x)

    def parse_graph_operation(self, serialized: str):
        kind, codomain, domain = serialized.split(":", 2)
        in_ids = list(map(int, codomain.split("x")))
        out_ids = list(map(int, domain.split("x")))
        return ir.Operation(kind, in_ids, out_ids)

    # Additional attributes
    def parse_graph_operations(self, serialized: str):
        return [self.parse_graph_operation(op) for op in serialized.split(";")]

    @staticmethod
    def parse_data_formats(formats: str):
        format_info: Dict[str, List[str]] = {}
        parts = formats.strip().split()
        for part in parts:
            key, values = part.split(":", 1)
            format_info[key] = values.split(";")
        return ir.DataFormats(**format_info)

    @classmethod
    def parse_tensor(cls, tensor: str):
        type_and_dt, id, layout_type, property, dims, arg = tensor.split(":")
        in_out_num, dt = type_and_dt.split("_")
        if in_out_num.startswith("in"):
            tensor_type = ir.TensorType.INPUT
            tensor_id = int(in_out_num[2:])
        elif in_out_num.startswith("out"):
            tensor_type = ir.TensorType.OUTPUT
            tensor_id = int(in_out_num[3:])
        else:
            raise ParseError(
                f"Logical tensor {tensor} should start with in/out"
            )

        def as_concrete_tensor(cls, **kwargs):
            return cls(
                id=int(id),
                dt=dt,
                type=tensor_type,
                type_id=tensor_id,
                property=ir.TensorProperty(property),
                shape=list(map(int, dims.split("x"))),
                **kwargs,
            )

        if layout_type == "strided":
            strides = list(map(int, arg.split("s")))
            return as_concrete_tensor(ir.StridedTensor, strides=strides)
        elif layout_type == "opaque":
            return as_concrete_tensor(ir.OpaqueTensor, layout_id=int(arg))
        elif layout_type == "any":
            return as_concrete_tensor(ir.AnyTensor)
        else:
            raise ParseError(f"Unexpected layout type {layout_type}")

    @classmethod
    def parse_tensors(cls, tensors: str):
        try:
            return list(map(cls.parse_tensor, tensors.strip().split()))
        except ValueError:
            raise ParseError(f"Could not parse logical tensors {tensors}")

    @classmethod
    def parse_graph_attrs(cls, attrs: str):
        spec = ParseSpec(attrs)
        fpmath_mode = "undef"
        if spec.read_literal("fpm:"):
            fpmath_mode = spec.buf
        return ir.GraphAttributes(fpmath=fpmath_mode)

    parse_acc_mode = identity
    parse_deterministic = identity
    parse_scratchpad_mode = identity

    # Additional template components
    parse_operation = identity
    parse_prim_kind = identity
    parse_partition_kind = identity
    parse_prop_kind = identity
    parse_engine = identity
    parse_impl = identity
    parse_backend = identity
    parse_shapes = identity
    parse_partition_id = staticmethod(int)
    parse_time = staticmethod(float)
    parse_timestamp = staticmethod(float)

    def primitive_verbose_map(self):
        return {
            "operation": ("operation", self.parse_operation),
            "engine": ("engine", self.parse_engine),
            "primitive": ("prim_kind", self.parse_prim_kind),
            "implementation": ("impl", self.parse_impl),
            "prop_kind": ("prop_kind", self.parse_prop_kind),
            "memory_descriptors": ("mds", self.parse_mds),
            "attributes": ("exts", self.parse_primitive_attrs),
            "auxiliary": ("aux", self.parse_aux),
            "problem_desc": ("shapes", self.parse_shapes),
            "exec_time": ("time", self.parse_time),
            "timestamp": ("timestamp", self.parse_timestamp),
        }

    def graph_verbose_map(self):
        return {
            "operation": ("operation", self.parse_operation),
            "engine": ("engine", self.parse_engine),
            "partition_id": ("partition_id", self.parse_partition_id),
            "partition_kind": ("partition_kind", self.parse_partition_kind),
            "operations": ("operations", self.parse_graph_operations),
            "data_formats": ("data_formats", self.parse_data_formats),
            "logical_tensors": ("tensors", self.parse_tensors),
            "fpmath_mode": ("exts", self.parse_graph_attrs),
            "implementation": ("impl", self.parse_impl),
            "backend": ("backend", self.parse_backend),
            "exec_time": ("time", self.parse_time),
            "timestamp": ("timestamp", self.parse_timestamp),
        }

    def _parse(self, line: str, template: str, mapping):
        for field, value in zip(template.split(","), line.split(",")):
            if field not in mapping:
                continue
            key, parse = mapping[field]
            try:
                yield key, parse(value)
            except ParseError:
                raise ParseError(f"parsing entry error: {field}: {value}")
            except ValueError as e:
                raise ParseError(f"parse error: {line} ({e!s})")

    def parse_primitive(self, line: str, template: Optional[str]):
        if template is None:
            template = self.default_primitive_template
        parsed = self._parse(line, template, self.primitive_verbose_map())
        return dict(parsed), ir.PrimitiveEntry

    def parse_graph(self, line: str, template: Optional[str]):
        if template is None:
            template = self.default_graph_template
        entry = dict(self._parse(line, template, self.graph_verbose_map()))
        data_formats = entry["data_formats"]
        operations = entry["operations"]
        zipped = zip(operations, data_formats.data, data_formats.filter)
        for operation, data, filter in zipped:
            operation.data = data or None
            operation.filter = filter or None
        del entry["data_formats"]
        return entry, ir.GraphEntry


def register(*, version: int):
    def registrar(impl: type):
        ParserImpl._impl_map[version] = impl
        return impl

    return registrar


@register(version=0)
class LegacyParserImpl(ParserImpl):
    pass


@register(version=1)
class V1ParserImpl(ParserImpl):
    def parse_md(self, descriptor):
        fields = descriptor.split(":")
        return ir.MemoryDescriptor(
            arg=fields[0],
            data_type=fields[1],
            properties=fields[2],
            format_kind=fields[3],
            tag=fields[4],
            strides=fields[5],
            flags=self.parse_md_flags(fields[6], fields[7:]),
        )


class Parser:
    _parser_impls: Dict[int, ParserImpl] = {}
    _default_events = "exec", "create", "create_nested"

    def __init__(
        self,
        input: Iterable[str],
        events: Optional[Iterable[str]] = None,
        components: Optional[Iterable[Component]] = None,
        error_handler: ContextManager = nullcontext(),
    ):
        if events is None:
            events = "exec", "create"
        if components is None:
            components = (Component.PRIMITIVE,)
        self.input = input
        self.events = set(events)
        self.components = set(components)
        self.error_handler = error_handler

    def _fix_template(self, template) -> Optional[str]:
        return template

    @staticmethod
    def _parse_leading_fields(input: Iterable[str]):
        MARKER = "onednn_verbose"
        for line in map(str.rstrip, input):
            if not line.startswith(f"{MARKER},"):
                continue
            try:
                _, operation, args = line.split(",", 2)
            except ValueError:
                continue
            component = Component.PRIMITIVE
            version = 0
            if operation.startswith("v"):
                try:
                    version = int(operation[1:])
                except ValueError:
                    pass
                else:
                    operation, args = args.split(",", 1)
            timestamp = None
            try:
                timestamp = float(operation)
            except ValueError:
                pass
            else:
                operation, args = args.split(",", 1)
            if operation in ("graph", "primitive", "ukernel"):
                if operation == "graph":
                    component = Component.GRAPH
                # else: use the default (Component.PRIMITIVE)
                operation, args = args.split(",", 1)
            yield line, version, timestamp, component, operation, args

    def __iter__(self) -> Iterator[Tuple[str, ir.HashableEntry]]:
        template = None
        cache: Dict[str, Tuple[dict, type]] = {}
        errors: Set[str] = set()
        parsed = self._parse_leading_fields(self.input)
        for line, version, timestamp, component, operation, args in parsed:
            if component == "graph":
                continue
            event = operation.split(":", 1)[0]
            if event == "info":
                for marker in ("template", "prim_template"):
                    if not args.startswith(f"{marker}:"):
                        continue
                    fixed_template = self._fix_template(args[len(marker) + 1 :])
                    if fixed_template is not None:
                        break
                else:
                    continue
                first_component, rest = fixed_template.split(",", 1)
                # Timestamp is usually out of order with respect to the
                # template because of missing component for "graph",
                # "primitive", "ukernel", etc.
                if first_component == "timestamp":
                    fixed_template = rest
                if template != fixed_template:
                    template = fixed_template
                    cache.clear()
                continue
            if event not in self.events:
                continue
            if component not in self.components:
                continue
            leading_args, last_arg = args.rsplit(",", 1)
            try:
                time = float(last_arg)
            except ValueError:
                time = 0.0
                leading_args = args
            key = f"v{version},{component!s},{operation},{leading_args}"
            if key in errors:
                continue
            success = False
            with self.error_handler:
                if key in cache:
                    params, container = cache[key]
                    params.update(time=time, timestamp=timestamp)
                    yield line, container(**params)
                new_line = f"{operation},{args}"
                params, container = self.parse(
                    new_line,
                    component,
                    template,
                    version,
                )
                cache[key] = params, container
                if timestamp is not None:
                    params.update(timestamp=timestamp)
                yield line, container(**params)
                success = True
            if not success:
                errors.add(key)

    def items(self) -> Iterable[Tuple[int, Tuple[str, ir.HashableEntry]]]:
        yield from enumerate(self)

    @staticmethod
    def _get_impl(version: int = 0) -> ParserImpl:
        if version in Parser._parser_impls:
            pass
        elif version in ParserImpl._impl_map:
            Parser._parser_impls[version] = ParserImpl._impl_map[version]()
        else:
            raise ParseError(f"No parser registered for version {version}.")
        return Parser._parser_impls[version]

    def parse(
        self,
        line: str,
        component: Component,
        template: Optional[str],
        version: int = 0,
    ):
        impl = self._get_impl(version)
        parser_map = {
            Component.GRAPH: impl.parse_graph,
            Component.PRIMITIVE: impl.parse_primitive,
        }
        parser = parser_map[component]
        return parser(line, template)
