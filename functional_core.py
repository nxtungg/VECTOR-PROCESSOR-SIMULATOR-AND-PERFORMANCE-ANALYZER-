"""Lõi tính toán thuần (functional core) dùng chung cho các simulator.

Tách phần "tính kết quả" ra khỏi phần "tính timing" để OoO simulator có thể
tự thực thi (thay vì delegate sang PipelineSimulator) và để tránh lặp code.

Quy ước:
- Operand vector là list[float].
- VREDUCE_* và VDOT trả về scalar (float).
- Các phép còn lại trả về vector (list[float]).

LƯU Ý semantics: đây là mô hình ISA GIÁO DỤC. Mục tiêu là đúng đắn và an toàn
(không bug silent), không phải tái hiện chính xác phần cứng. Các điểm cần biết:
- Phép logic ép float → int trước khi thao tác bit (xem ``logical``).
- ``mask_threshold`` dùng heuristic threshold = trung bình, KHÔNG phải mask phần
  cứng thực (phần cứng dùng predicate register / so sánh tường minh).
"""

from typing import Dict, List, Tuple, Union, Optional

Number = float
Vector = List[float]
Result = Union[Vector, Number]


# Các opcode trả về scalar thay vì vector
SCALAR_RESULT_OPCODES = {"VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN", "VDOT"}

# Phân nhóm opcode để validate trong compute()
_BINARY_OPCODES = {"VADD", "VSUB", "VMUL", "VDIV", "VMAX", "VMIN"}
_SCALAR_OPCODES = {"VADDS", "VSUBS", "VMULS", "VDIVS"}
_LOGICAL_OPCODES = {"VAND", "VOR", "VXOR"}
_REDUCE_OPCODES = {"VREDUCE_SUM", "VREDUCE_MAX", "VREDUCE_MIN"}


def is_scalar_result(opcode: str) -> bool:
    """Lệnh có kết quả là scalar (ghi vào scalar register) hay không."""
    return opcode in SCALAR_RESULT_OPCODES


def _condition(name: str):
    cond_map = {
        "gt": lambda x, y: x > y,
        "lt": lambda x, y: x < y,
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "ge": lambda x, y: x >= y,
        "le": lambda x, y: x <= y,
    }
    if name not in cond_map:
        raise ValueError(f"Unknown condition: {name}")
    return cond_map[name]


def _check_lengths(opcode: str, a: Vector, b: Vector) -> None:
    """Kiểm tra hai operand vector cùng độ dài.

    Tránh dùng zip() cắt ngầm khi độ dài lệch nhau (bug silent). Nếu không khớp
    thì raise ValueError kèm opcode và độ dài để dễ debug.
    """
    if len(a) != len(b):
        raise ValueError(
            f"{opcode}: vector length mismatch ({len(a)} vs {len(b)}); "
            f"hai operand phải cùng độ dài"
        )


class FunctionalCore:
    """Thực thi phần chức năng (giá trị) của một lệnh vector.

    Stateless cho phần lớn opcode; chỉ giữ ``mask_registers`` để hỗ trợ các
    lệnh masked. Caller cấp operand đã đọc sẵn để core không phụ thuộc vào
    register file hay memory cụ thể.
    """

    def __init__(self, scalars: Optional[Dict[str, float]] = None):
        self.scalars: Dict[str, float] = dict(scalars or {})
        self.mask_registers: Dict[str, List[bool]] = {}

    # ===== Số học hai toán hạng vector =====

    def binary(self, opcode: str, a: Vector, b: Vector) -> Vector:
        _check_lengths(opcode, a, b)
        if opcode == "VADD":
            return [x + y for x, y in zip(a, b)]
        if opcode == "VSUB":
            return [x - y for x, y in zip(a, b)]
        if opcode == "VMUL":
            return [x * y for x, y in zip(a, b)]
        if opcode == "VDIV":
            return [x / y if y != 0 else 0 for x, y in zip(a, b)]
        if opcode == "VMAX":
            return [max(x, y) for x, y in zip(a, b)]
        if opcode == "VMIN":
            return [min(x, y) for x, y in zip(a, b)]
        raise NotImplementedError(f"binary: unsupported opcode {opcode!r}")

    # ===== Số học với scalar =====

    def scalar_op(self, opcode: str, a: Vector, scalar_token: str) -> Vector:
        s = self.resolve_scalar(scalar_token)
        if opcode == "VADDS":
            return [x + s for x in a]
        if opcode == "VSUBS":
            return [x - s for x in a]
        if opcode == "VMULS":
            return [x * s for x in a]
        if opcode == "VDIVS":
            return [x / s if s != 0 else 0 for x in a]
        raise NotImplementedError(f"scalar_op: unsupported opcode {opcode!r}")

    # ===== Logical =====

    def logical(self, opcode: str, a: Vector, b: Vector) -> Vector:
        # Mô hình ISA GIÁO DỤC: phép bit chỉ định nghĩa trên số nguyên, nên ta
        # ép float → int trước khi thao tác bit rồi đưa kết quả về float để giữ
        # nhất quán kiểu dữ liệu vector. Phần cứng thực dùng biểu diễn bit gốc;
        # ở đây int(x) cắt phần thập phân (truncate toward zero).
        _check_lengths(opcode, a, b)
        ai = [int(x) for x in a]
        bi = [int(x) for x in b]
        if opcode == "VAND":
            res = [x & y for x, y in zip(ai, bi)]
        elif opcode == "VOR":
            res = [x | y for x, y in zip(ai, bi)]
        elif opcode == "VXOR":
            res = [x ^ y for x, y in zip(ai, bi)]
        else:
            raise NotImplementedError(f"logical: unsupported opcode {opcode!r}")
        return [float(x) for x in res]

    # ===== Reduction / dot (trả scalar) =====

    def reduction(self, opcode: str, a: Vector) -> Number:
        if opcode == "VREDUCE_SUM":
            return sum(a)
        if opcode == "VREDUCE_MAX":
            return max(a) if a else 0
        if opcode == "VREDUCE_MIN":
            return min(a) if a else 0
        raise NotImplementedError(f"reduction: unsupported opcode {opcode!r}")

    def dot(self, a: Vector, b: Vector) -> Number:
        _check_lengths("VDOT", a, b)
        return sum(x * y for x, y in zip(a, b))

    # ===== Mask =====

    def mask_threshold(self, src: Vector, condition: str) -> List[bool]:
        # HEURISTIC giáo dục: threshold = trung bình các phần tử. ĐÂY KHÔNG phải
        # mask phần cứng thực (phần cứng so sánh với giá trị/predicate tường
        # minh, không tự suy ra ngưỡng từ dữ liệu).
        threshold = sum(src) / len(src) if src else 0
        cmp = _condition(condition)
        return [cmp(v, threshold) for v in src]

    def mask_compare(self, a: Vector, b: Vector, condition: str) -> List[bool]:
        _check_lengths("VMASK_COND", a, b)
        cmp = _condition(condition)
        return [cmp(x, y) for x, y in zip(a, b)]

    def masked_binary(self, opcode: str, a: Vector, b: Vector,
                    mask: List[bool]) -> Vector:
        _check_lengths(opcode, a, b)
        if len(mask) != len(a):
            raise ValueError(
                f"{opcode}: mask length mismatch ({len(mask)} vs {len(a)})"
            )
        if opcode == "VADD_MASKED":
            return [x + y if m else x for x, y, m in zip(a, b, mask)]
        if opcode == "VSUB_MASKED":
            return [x - y if m else x for x, y, m in zip(a, b, mask)]
        raise NotImplementedError(f"masked_binary: unsupported opcode {opcode!r}")

    def masked_move(self, src: Vector, existing: Vector, mask: List[bool]) -> Vector:
        if not (len(src) == len(existing) == len(mask)):
            raise ValueError(
                f"VMOV_MASKED: length mismatch (src={len(src)}, "
                f"existing={len(existing)}, mask={len(mask)})"
            )
        return [s if m else e for s, e, m in zip(src, existing, mask)]

    # ===== Tiện ích =====

    def resolve_scalar(self, token: str) -> float:
        """Lấy giá trị scalar: từ bảng scalars nếu là biến, ngược lại parse số.

        Parse float an toàn: nếu token không phải số hợp lệ thì raise ValueError
        rõ ràng thay vì để float() ném lỗi khó hiểu / crash ngầm.
        """
        if token in self.scalars:
            return self.scalars[token]
        try:
            return float(token)
        except (TypeError, ValueError):
            raise ValueError(
                f"resolve_scalar: invalid scalar operand {token!r}; "
                f"không phải biến scalar đã biết cũng không phải số hợp lệ"
            )

    def compute(self, opcode: str, operands: List[Vector],
                extra: Optional[str] = None) -> Tuple[str, Result]:
        """Thực thi opcode số học/logic/reduction trên các operand đã đọc.

        Validate opcode hợp lệ, số lượng toán hạng và độ dài vector TRƯỚC khi
        thực thi để tránh bug silent (zip cắt ngầm, thiếu operand...).

        Args:
            opcode: mã lệnh.
            operands: danh sách vector toán hạng (đã đọc từ register file).
            extra: tham số phụ (scalar token cho *S).

        Returns:
            ("vector", list) hoặc ("scalar", float).
        """
        # Validate opcode hợp lệ
        valid = (_BINARY_OPCODES | _SCALAR_OPCODES | _LOGICAL_OPCODES
                | _REDUCE_OPCODES | {"VDOT"})
        if opcode not in valid:
            raise ValueError(
                f"compute: unsupported opcode {opcode!r}; "
                f"hợp lệ: {sorted(valid)}"
            )

        def _need(n: int) -> None:
            if len(operands) < n:
                raise ValueError(
                    f"{opcode}: cần {n} operand nhưng nhận {len(operands)}"
                )

        if opcode in _BINARY_OPCODES:
            _need(2)
            _check_lengths(opcode, operands[0], operands[1])
            return "vector", self.binary(opcode, operands[0], operands[1])

        if opcode in _SCALAR_OPCODES:
            _need(1)
            if extra is None:
                raise ValueError(f"{opcode}: thiếu toán hạng scalar (extra)")
            return "vector", self.scalar_op(opcode, operands[0], extra)

        if opcode in _LOGICAL_OPCODES:
            _need(2)
            _check_lengths(opcode, operands[0], operands[1])
            return "vector", self.logical(opcode, operands[0], operands[1])

        if opcode in _REDUCE_OPCODES:
            _need(1)
            return "scalar", self.reduction(opcode, operands[0])

        if opcode == "VDOT":
            _need(2)
            _check_lengths(opcode, operands[0], operands[1])
            return "scalar", self.dot(operands[0], operands[1])

        # Không tới được do đã validate ở trên
        raise NotImplementedError(f"compute: unsupported opcode {opcode!r}")
