"""Minimal GGUF v3 reader: KV metadata + tensor directory with absolute offsets.

Used by the expert-major converter. Reads only the header; never touches tensor
data unless a caller explicitly requests a tensor slice.
"""
import struct

# ggml type -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
          6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Tensor:
    __slots__ = ("name", "dims", "type_id", "type", "rel_offset", "abs_offset",
                 "nbytes", "nelem")

    def __repr__(self):
        return (f"<{self.name} {self.type} {self.dims} "
                f"@{self.abs_offset} {self.nbytes}B>")


class GGUF:
    def __init__(self, path):
        self.path = path
        self.kv = {}
        self.tensors = {}
        self._parse()

    def _parse(self):
        f = open(self.path, "rb")
        self.f = f

        def rd(n):
            b = f.read(n)
            if len(b) != n:
                raise EOFError("truncated GGUF header")
            return b

        def u32():
            return struct.unpack("<I", rd(4))[0]

        def u64():
            return struct.unpack("<Q", rd(8))[0]

        def val(t):
            if t == 8:
                return rd(u64()).decode("utf-8", "replace")
            if t == 9:
                et, n = u32(), u64()
                if et in _FIXED and n > 64:
                    f.seek(struct.calcsize(_FIXED[et]) * n, 1)
                    return f"<{n} skipped>"
                return [val(et) for _ in range(n)]
            return struct.unpack(_FIXED[t], rd(struct.calcsize(_FIXED[t])))[0]

        if rd(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.version = u32()
        n_tensors, n_kv = u64(), u64()
        for _ in range(n_kv):
            k = rd(u64()).decode("utf-8", "replace")
            self.kv[k] = val(u32())

        raw = []
        for _ in range(n_tensors):
            t = Tensor()
            t.name = rd(u64()).decode("utf-8", "replace")
            t.dims = [u64() for _ in range(u32())]
            t.type_id = u32()
            t.rel_offset = u64()
            raw.append(t)

        # Data section starts after the header, aligned up.
        align = self.kv.get("general.alignment", 32)
        pos = f.tell()
        self.data_start = (pos + align - 1) // align * align

        for t in raw:
            name, blk, bb = GGML_TYPES.get(t.type_id, (f"T{t.type_id}", 1, 4))
            t.type = name
            n = 1
            for d in t.dims:
                n *= d
            t.nelem = n
            if n % blk:
                raise ValueError(f"{t.name}: {n} elems not a multiple of block {blk}")
            t.nbytes = n // blk * bb
            t.abs_offset = self.data_start + t.rel_offset
            self.tensors[t.name] = t

    def expert_slice(self, tensor_name, expert):
        """Absolute (offset, nbytes) of one expert inside a 3-D MoE tensor.

        MoE expert tensors are [ne0, ne1, n_experts]; expert e occupies a
        contiguous ne0*ne1 slice. Verified to land on block boundaries.
        """
        t = self.tensors[tensor_name]
        if len(t.dims) != 3:
            raise ValueError(f"{tensor_name} is not a 3-D expert tensor")
        n_experts = t.dims[2]
        if not 0 <= expert < n_experts:
            raise IndexError(expert)
        stride = t.nbytes // n_experts
        if t.nbytes % n_experts:
            raise ValueError(f"{tensor_name}: expert stride not byte-exact")
        return t.abs_offset + stride * expert, stride
