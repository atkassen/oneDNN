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
from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import MISSING, dataclass, field, fields
from typing import Dict, List, Optional, Union


def alias(attr):
    def getter(self):
        return getattr(self, attr)

    def setter(self, value):
        return setattr(self, attr, value)

    def deleter(self):
        return delattr(self, attr)

    return property(getter, setter, deleter, attr)


def hash_str(obj):
    return getattr(obj.__class__, "__hash_str__", str)(obj)


@dataclass(eq=False)
class Mapping(MutableMapping):
    def __getitem__(self, item):
        try:
            value = getattr(self, item)
            if isinstance(value, int):
                value = str(value)
            elif isinstance(value, float):
                value = str(value)
                # The verbose converter assumes defaults are 1.0, whereas
                # oneDNN assumes defaults are 0.0. This is a workaround so that
                # we don't accidentally drop these values, instead setting as 0
                # or 1 which will always be sent through to the benchdnn
                # reproducer
                if value[-2:] == ".0":
                    value = value[:-2]
            return value
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def __delitem__(self, item):
        delattr(self, item)

    def __len__(self):
        return len(fields(self))

    def __iter__(self):
        for datafield in fields(self):
            yield datafield.name

    def __hash__(self):
        return hash(hash_str(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return hash_str(self) == hash_str(other)

    def __str__(self):
        raise NotImplementedError

    def __hash_str__(self):
        return str(self)

    def __repr__(self):
        child_reprs = []
        for key, value in self.items():
            child_reprs.append(f"{key!r}: {value!r}")
        return "{" + ", ".join(child_reprs) + "}"


@dataclass(eq=False)
class MemoryDescriptor(Mapping):
    @dataclass(eq=False)
    class Flags(Mapping):
        value: str
        s8_comp_mask: Optional[str] = None
        zp_comp_mask: Optional[str] = None
        scale_adjust: float = 1.0

        def __str__(self):
            my_str = self.value
            if self.s8_comp_mask is not None:
                my_str += f":s8m{self.s8_comp_mask}"
            if self.zp_comp_mask is not None:
                my_str += f":s8m{self.zp_comp_mask}"
            if self.scale_adjust != 1.0:
                my_str += f":sa{self.scale_adjust}"
            return my_str

    arg: str
    data_type: str
    properties: str
    format_kind: str
    tag: str
    flags: Flags
    strides: str = ""  # Pre-v3.1 does not have strides

    padding = alias("properties")

    def __len__(self):
        return 1 + super().__len__()

    def __iter__(self):
        yield from super().__iter__()
        yield "padding"

    def _format(self, tag: str, convert) -> str:
        header = f"{self.arg}:{self.data_type}"
        return ":".join(
            [
                header,
                self.properties,
                self.format_kind,
                tag,
                self.strides,
                convert(self.flags),
            ]
        )

    def __str__(self):
        return self._format(self.tag, str)

    def __hash_str__(self):
        tag = self.tag
        if "a" not in self.properties:
            return self._format(tag, hash_str)
        for i, c in enumerate(tag):
            if not c.isalpha():
                return self._format(string.ascii_lowercase[:i], hash_str)
        return self._format(string.ascii_lowercase[: len(tag)], hash_str)


@dataclass(eq=False)
class Dropout(Mapping):
    tag: Optional[str] = None

    def __str__(self):
        return self.tag or ""


class FormattedMapping(Mapping):
    @abstractmethod
    def _format(self, _) -> str:
        raise NotImplementedError

    def __str__(self):
        return self._format(str)

    def __hash_str__(self):
        return self._format(hash_str)


@dataclass(eq=False)
class PostOp(FormattedMapping):
    alg: str

    def _format(self, convert):
        required_args = []
        optional_args = []
        seen_non_default = False
        for datafield in reversed(fields(self)):
            if datafield.name == "alg":
                continue
            value = getattr(self, datafield.name)
            if datafield.default is MISSING:
                required_args.append(value)
                continue
            if not seen_non_default and value == datafield.default:
                continue
            seen_non_default = True
            optional_args.append(value)
        args = [self.alg] + required_args[::-1] + optional_args[::-1]
        return ":".join(map(convert, args))


@dataclass(eq=False)
class SumPostOp(PostOp):
    alg: str = "sum"
    scale: float = 1.0
    zp: int = 0
    dt: str = ""


@dataclass(eq=False)
class DepthwiseScales(Mapping):
    mask: int = 0
    value: Optional[str] = None

    def __str__(self):
        if self.value is not None:
            return f"{self.mask}:{self.value}"
        if self.mask != 0:
            return str(self.mask)
        return ""


@dataclass(eq=False)
class KSPMixin:
    ksp: str


@dataclass(eq=False)
class DepthwisePostOp(PostOp, KSPMixin):
    alg: str = "dw"
    dst_dt: str = "f32"
    wei_dt: str = "f32"
    scales: DepthwiseScales = DepthwiseScales()

    def __len__(self):
        return 1 + super().__len__()

    def __iter__(self):
        yield "alg"
        yield from super().__iter__()


@dataclass(eq=False)
class PreLUPostOp(PostOp):
    alg: str = "prelu"
    mask: int = 0
    has_scaleshift: bool = False

    def __getitem__(self, item):
        if item == "has_scaleshift":
            return "true" if self.has_scaleshift else ""
        return super().__getitem__(item)

    def __str__(self):
        if self.has_scaleshift:
            return f"{self.alg}:{self.mask}:true"
        return f"{self.alg}:{self.mask}"


@dataclass(eq=False)
class EltwisePostOp(PostOp):
    alpha: float = 0.0
    beta: float = 0.0
    scale: float = 1.0


@dataclass(eq=False)
class BinaryPostOp(PostOp):
    dt: str
    mask: int = 0
    tag: str = "any"


@dataclass(eq=False)
class QuantizationParam(Mapping):
    value: float
    data_type: str
    mask: int = 0
    groups: str = ""

    def __str__(self):
        if self.groups:
            return f"{self.mask}:{self.data_type}:{self.groups}"
        return f"{self.mask}:{self.data_type}"


@dataclass(eq=False)
class Scale(QuantizationParam):
    value: float = 1.0
    data_type: str = "f32"


@dataclass(eq=False)
class ZeroPoint(QuantizationParam):
    value: int = 0
    data_type: str = "s32"


class CompositeAttribute:
    def __str__(self):
        raise NotImplementedError


@dataclass(eq=False)
class FPMathMode(CompositeAttribute):
    mode: str
    apply_to_int: bool = False

    def __str__(self):
        a2i_str = ":true" if self.apply_to_int else ""
        return self.mode + a2i_str


class RoundingMode(CompositeAttribute, enum.Enum):
    ENVIRONMENT = "environment"
    STOCHASTIC = "stochastic"

    def __str__(self):
        return self.value


PrimitiveAttribute = Union[
    str,  # acc-mode, etc
    FPMathMode,
    Dropout,
    List[PostOp],
    Dict[str, Scale],
    Dict[str, ZeroPoint],
    Dict[str, RoundingMode],
    Scale,  # oscale
]


@dataclass(eq=False)
class PrimitiveAttributes(FormattedMapping):
    acc_mode: Optional[str] = None
    deterministic: Optional[str] = None
    dropout: Optional[Dropout] = None
    fpmath: Optional[FPMathMode] = None
    oscale: Optional[Scale] = None
    post_ops: Optional[List[PostOp]] = None
    rounding_mode: Optional[Dict[str, RoundingMode]] = None
    scales: Optional[Dict[str, Scale]] = None
    scratchpad: Optional[str] = None
    zero_points: Optional[Dict[str, ZeroPoint]] = None

    acc = alias("acc_mode")

    @staticmethod
    def _field_name_to_attr_name(field_name: str):
        return "attr-" + field_name.replace("_", "-")

    def _attr_name_to_field_name(self, item: str):
        original_item = item
        for f in fields(self):
            if item == self._field_name_to_attr_name(f.name):
                return f.name
        raise KeyError(original_item)

    def __getitem__(self, item: str):
        value = getattr(self, self._attr_name_to_field_name(item))
        if value is None:
            raise KeyError(item)
        return value

    def __setitem__(self, item: str, value: PrimitiveAttribute):
        return setattr(self, self._attr_name_to_field_name(item), value)

    def __delitem__(self, item: str):
        setattr(self, self._attr_name_to_field_name(item), None)

    def __iter__(self):
        for f in fields(self):
            if getattr(self, f.name) is not None:
                yield self._field_name_to_attr_name(f.name)

    def __len__(self):
        return len(list(iter(self)))

    def _format(self, convert):
        parts = []
        for key, attr in self.items():
            if isinstance(attr, list):
                sub_parts = "+".join(map(convert, attr))
                parts.append(f"{key}:{sub_parts}")
            elif isinstance(attr, dict):
                converted = (f"{k}:{convert(v)}" for k, v in attr.items())
                combined = "+".join(converted)
                parts.append(f"{key}:{combined}")
            else:
                parts.append(f"{key}:{convert(attr)}")
        return " ".join(parts)


class TensorType(enum.Enum):
    INPUT = "in"
    OUTPUT = "out"

    def __str__(self):
        return self.value


class TensorProperty(enum.Enum):
    UNDEF = "undef"
    VARIABLE = "variable"
    CONSTANT = "constant"

    def __str__(self):
        return self.value


@dataclass(eq=False)
class DataFormats(Mapping):
    data: List[str] = field(default_factory=list)
    filter: List[str] = field(default_factory=list)


@dataclass(eq=False)
class Operation(Mapping):
    kind: str
    inputs: List[int]
    outputs: List[int]
    data: Optional[str] = None
    filter: Optional[str] = None

    def __str__(self):
        ins = "x".join(f"t{id}" for id in self.inputs)
        outs = "x".join(f"t{id}" for id in self.outputs)
        return f"{self.kind}:{ins}->{outs}"


@dataclass(eq=False)
class Tensor(Mapping):
    type: TensorType
    type_id: int
    dt: str
    id: int
    property: str
    shape: List[int]
    layout_type: str

    def __str__(self):
        shape = "x".join(map(str, self.shape))
        return (
            f"{self.type!s}{self.type_id}_{self.dt}:{self.id}:"
            + f"{self.layout_type}:{self.property}:{shape}"
        )


@dataclass(eq=False)
class StridedMixin:
    strides: List[int]


@dataclass(eq=False)
class StridedTensor(Tensor, StridedMixin):
    layout_type: str = "strided"

    def __str__(self):
        strides = "s".join(map(str, self.strides))
        return f"{super().__str__()}:{strides}"


@dataclass(eq=False)
class OpaqueMixin:
    layout_id: int


@dataclass(eq=False)
class OpaqueTensor(Tensor, OpaqueMixin):
    layout_type: str = "opaque"

    def __str__(self):
        return f"{super().__str__()}:{self.layout_id}"


@dataclass(eq=False)
class AnyTensor(Tensor):
    layout_type: str = "any"

    def __str__(self):
        return f"{super().__str__()}:any"


@dataclass(eq=False)
class GraphAttributes(Mapping):
    fpmath: str


@dataclass(eq=False)
class HashableEntry(FormattedMapping):
    operation: str
    engine: str
    impl: str
    component: str

    def _format(self, _):
        return f"{self.component},{self.operation},{self.engine}"


@dataclass(eq=False)
class PrimitiveMixin:
    prim_kind: str
    prop_kind: str
    aux: Dict[str, str]
    mds: List[MemoryDescriptor]
    shapes: str
    exts: PrimitiveAttributes


@dataclass(eq=False)
class HashablePrimitiveEntry(HashableEntry, PrimitiveMixin):
    component: str = "primitive"

    def _format(self, convert):
        parts = [
            super()._format(convert),
            self.prim_kind,
            self.impl,
            self.prop_kind,
            " ".join(map(convert, self.mds)),
            convert(self.exts),
            " ".join(f"{k}:{convert(v)}" for k, v in self.aux.items()),
            self.shapes,
        ]
        return ",".join(parts)


@dataclass(eq=False)
class GraphMixin:
    partition_id: int
    partition_kind: str
    backend: str
    operations: List[Operation]
    tensors: List[Tensor]
    exts: GraphAttributes


@dataclass(eq=False)
class HashableGraphEntry(HashableEntry, GraphMixin):
    component: str = "graph"

    def _get_data_formats(self):
        data, filter = [], []
        have_data = False
        have_filter = False
        for op in self.operations:
            data.append(op.data or "")
            filter.append(op.filter or "")
            have_data |= op.data is not None
            have_filter |= op.filter is not None
        parts = []
        if have_data:
            parts.append(";".join(data))
        if have_filter:
            parts.append(";".join(filter))
        return " ".join(parts)

    def _format(self, convert):
        parts = [
            super()._format(convert),
            str(self.partition_id),
            self.partition_kind,
            ";".join(map(str, self.operations)),
            self._get_data_formats(),
            " ".join(map(str, self.tensors)),
            f"fpm:{self.exts.fpmath}" if self.exts.fpmath else "",
            self.impl,
            self.backend,
        ]
        return ",".join(parts)


class VerboseExtrasMixin:
    def __init__(
        self,
        *,
        time: float = 0.0,
        timestamp: Optional[float] = None,
        version: int = 0,
        **kwargs,
    ):
        self.time = time
        self.timestamp = timestamp
        self.version = version
        super().__init__(**kwargs)

    def __str__(self):
        return f"onednn_verbose,v{self.version},{super().__str__()},{self.time}"


class PrimitiveEntry(VerboseExtrasMixin, HashablePrimitiveEntry):
    pass


class GraphEntry(VerboseExtrasMixin, HashableGraphEntry):
    pass
