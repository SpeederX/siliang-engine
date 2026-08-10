"""Write a SINGLE self-contained GGUF with experts stored expert-major.

Replaces the experts.slab sidecar. The slab forces the original GGUF and a
duplicate copy of every expert to coexist. This writes one self-contained file
with the expert bytes reordered in place.

WHY REORDER AT ALL
------------------
GGUF packs all experts of one projection into a single 3-D tensor, so a layer's
gate/up/down live in separate regions. Loading one expert therefore needs
scattered reads. The expert-major layout makes that access contiguous; measure
its effect on the target model and storage before making a performance claim.

LAYOUT
------
Non-expert tensors are copied verbatim. Per layer, the expert data is written
into one packed byte tensor,

    [expert 0: gate | up | down | pad][expert 1: gate | up | down | pad] ...

so all projections for one expert are adjacent and one arena miss can be served
by one contiguous read. The original projection tensor directory entries are
replaced by one exact-size I8 byte tensor per MoE layer:

    blk.L.ffn_exps_packed.weight

The `siliangem.*` metadata records each logical projection's name, type, shape,
and byte length; the ordered lengths define each part's cumulative offset, and
the per-layer metadata records the padded expert stride.
At load time the patched loader removes the opaque packed entry from its weight
map and synthesises the original logical `ffn_*_exps.weight` tensors as strided
views into the packed region.

LIMITATION: those logical views depend on mmap. The patched loader refuses an
expert-major file when mmap is off; an ordinary contiguous copy of a strided
view would consume the wrong expert bytes.

STRIDE PADDING: expert_bytes is padded to lcm(part type sizes, 512) because
ggml-cuda computes s02 = nb[2]/type_size by integer division (mmvq.cu), so an
unpadded stride silently misaddresses every expert after the first; 512 is the
volume logical sector size that FILE_FLAG_NO_BUFFERING requires.

TWO GENERALITY FIXES THE SLAB FORMAT LACKED
-------------------------------------------
1. PER-LAYER STRIDES. The slab assumed one global expert stride. Real GGUFs
   can break that when selected layers use different projection quantization
   types. Each layer is internally uniform, so per-layer strides cover this.
2. FLEXIBLE PARTS. Gemma fuses gate and up into `ffn_gate_up_exps.weight`, so
   the part list cannot be hardcoded to three.

Metadata used to reconstruct the logical projection tensors:
    siliangem.expert_major   bool
    siliangem.n_experts      u32
    siliangem.part_names     comma-separated projection names
    siliangem.expert_bytes   [u32] per layer, the value for nb[2]
    siliangem.part_bytes     [u32] per (layer, part)
    siliangem.part_types     [u32] GGML type per (layer, part)
    siliangem.part_ne0       [u32] logical ne[0] per (layer, part)
    siliangem.part_ne1       [u32] logical ne[1] per (layer, part)
    siliangem.layer_index    [u32] actual source layer numbers
"""
import argparse
import os
import struct
import sys
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import GGUF, GGML_TYPES          # noqa: E402

# Candidate part groupings, most specific first. Gemma fuses gate and up.
PART_SETS = [
    ("gate", "up", "down"),
    ("gate_up", "down"),
]


def discover(g):
    """Return (part_names, n_layers, n_experts, per_layer_part_bytes).

    per_layer_part_bytes[layer][part] -> byte size of that expert-tensor's
    single-expert slice. Validates that experts within a layer are uniform,
    which they always are; only ACROSS layers do sizes differ.
    """
    for parts in PART_SETS:
        layers = {}
        ok = True
        for part in parts:
            suffix = f"ffn_{part}_exps.weight"
            found = {}
            for name, t in g.tensors.items():
                if name.endswith(suffix) and name.startswith("blk."):
                    layer = int(name.split(".")[1])
                    n_exp = t.dims[-1]
                    if t.nbytes % n_exp:
                        raise SystemExit(
                            f"{name}: {t.nbytes} B not divisible by {n_exp} experts")
                    found[layer] = (t.nbytes // n_exp, n_exp, name, t.type_id)
            if not found:
                ok = False
                break
            for layer, v in found.items():
                layers.setdefault(layer, {})[part] = v
        if not ok:
            continue
        complete = {L: d for L, d in layers.items() if len(d) == len(parts)}
        if not complete:
            continue
        n_experts = {v[1] for d in complete.values() for v in d.values()}
        if len(n_experts) != 1:
            raise SystemExit(f"inconsistent expert count across tensors: {n_experts}")
        return parts, sorted(complete), n_experts.pop(), complete
    raise SystemExit("no recognised expert tensors (tried %s)" % (PART_SETS,))


# ---- GGUF writing ---------------------------------------------------------

def w_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def w_kv_u32(k, v):
    return w_str(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def w_kv_bool(k, v):
    return w_str(k) + struct.pack("<I", 7) + struct.pack("<?", bool(v))


def w_kv_u64arr(k, vals):
    return (w_str(k) + struct.pack("<I", 9) + struct.pack("<I", 10)
            + struct.pack("<Q", len(vals))
            + b"".join(struct.pack("<Q", v) for v in vals))


def w_kv_str(k, v):
    return w_str(k) + struct.pack("<I", 8) + w_str(v)


def verify(a, g, parts, layers, n_experts, geom):
    """Byte-compare random (layer, expert, part) slices dst vs src.

    The reorder is pure data movement, so every byte must survive it. A wrong
    stride or a mis-seek can produce a file that still loads and generates
    subtly wrong text. Reading both sides independently provides the required
    byte-level check.
    """
    import random
    from gguf import GGUFReader

    rd = GGUFReader(a.dst)
    fld = rd.fields.get("siliangem.expert_bytes")
    if fld is None:
        raise SystemExit("dst is missing siliangem.expert_bytes - not an expert-major file")
    eb = [int(x) for x in fld.contents()]

    # Read through the packed tensor's own memmap: no offset arithmetic of our
    # own, since a bug in the CHECKER is indistinguishable from a corrupt file
    # (the first version of this function reported 64/64 false mismatches).
    packed = {}
    for t in rd.tensors:
        if ".ffn_exps_packed.weight" in t.name:
            packed[int(t.name.split(".")[1])] = t.data.view("uint8").reshape(-1)
    if len(packed) != len(layers):
        raise SystemExit(f"dst has {len(packed)} packed tensors, expected {len(layers)}")

    fs = open(a.src, "rb")
    random.seed(0xBEEF)
    bad = 0
    for _ in range(a.samples):
        L = random.choice(layers)
        e = random.randrange(n_experts)
        li = layers.index(L)
        buf = packed[L]
        off = e * eb[li]          # padded stride, exactly what nb[2] will hold
        for p in parts:
            per, _, tname, _tid = geom[L][p]
            got = buf[off:off + per].tobytes()
            fs.seek(g.tensors[tname].abs_offset + e * per)
            want = fs.read(per)
            off += per            # parts are contiguous within one expert
            if want != got:
                bad += 1
                first = next((i for i,(x,y) in enumerate(zip(want,got)) if x!=y), None)
                print(f"  MISMATCH layer {L} expert {e} part {p} (first differing byte {first})")
                break
    fs.close()
    if bad:
        raise SystemExit(f"verify FAILED: {bad}/{a.samples} sampled experts differ")
    print(f"verify: {a.samples}/{a.samples} random (layer,expert) slices "
          f"byte-identical to source  OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without writing")
    ap.add_argument("--verify", action="store_true",
                    help="byte-compare random experts in --dst against --src")
    ap.add_argument("--samples", type=int, default=64)
    a = ap.parse_args()

    if os.path.exists(a.dst) and not a.dry_run and not a.verify:
        raise SystemExit(f"refusing to overwrite existing {a.dst}")

    g = GGUF(a.src)
    parts, layers, n_experts, geom = discover(g)
    n_layers = len(layers)

    if a.verify:
        return verify(a, g, parts, layers, n_experts, geom)

    # ---- expert stride padding ------------------------------------------
    # TWO hard constraints, satisfied by one pad target.
    #
    # 1. ggml-cuda computes  s02 = src0->nb[2] / ggml_type_size(src0->type)
    #    by INTEGER DIVISION (mmvq.cu:1242). If the stride is not an exact
    #    multiple of the type size it truncates, and every expert after the
    #    first is read from the wrong offset -- fluent wrong output, no crash.
    #    A normal GGUF satisfies this by construction; a packed
    #    gate+up+down stride does not.
    #
    # 2. FILE_FLAG_NO_BUFFERING requires read offset, length and buffer
    #    address to be multiples of the VOLUME LOGICAL SECTOR SIZE. The public
    #    format contract uses 512 bytes; this is not the NTFS cluster size.
    #
    # So: pad to lcm(all part type sizes, 512).
    #
    # Why 512 and not a larger filesystem allocation unit: the direct-I/O
    # contract requires logical-sector alignment. Extra padding increases both
    # file size and bytes transferred without a portable correctness benefit.
    raw_expert_bytes = [sum(geom[L][p][0] for p in parts) for L in layers]
    part_bytes = [geom[L][p][0] for L in layers for p in parts]

    type_sizes = sorted({GGML_TYPES[geom[L][p][3]][2] for L in layers for p in parts})
    stride_lcm = 1
    for ts in type_sizes:
        stride_lcm = stride_lcm * ts // gcd(stride_lcm, ts)
    SECTOR = 512
    pad_target = stride_lcm * SECTOR // gcd(stride_lcm, SECTOR)

    expert_bytes = [((eb + pad_target - 1) // pad_target) * pad_target
                    for eb in raw_expert_bytes]
    packed_total = sum(eb * n_experts for eb in expert_bytes)
    raw_total = sum(eb * n_experts for eb in raw_expert_bytes)

    expert_names = {geom[L][p][2] for L in layers for p in parts}
    others = [t for n, t in g.tensors.items() if n not in expert_names]
    other_total = sum(t.nbytes for t in others)

    print(f"source        : {a.src}")
    print(f"destination   : {a.dst}")
    print(f"parts         : {','.join(parts)}")
    print(f"layers        : {n_layers}   experts/layer: {n_experts}")
    uniq = sorted(set(expert_bytes))
    print(f"expert stride : {len(uniq)} distinct "
          f"({', '.join(f'{u:,}B' for u in uniq[:4])}"
          f"{'...' if len(uniq) > 4 else ''})")
    if len(uniq) > 1:
        print("                -> per-layer strides in use; the old slab format "
              "could not express this")

    # ---- state the conversion's ongoing cost UP FRONT --------------------
    # Padding is not free: it can add bytes to every expert miss. Report the
    # byte cost before conversion; throughput impact must be measured on the
    # target model, storage device, and arena configuration.
    add = packed_total - raw_total
    pct = 100.0 * add / raw_total if raw_total else 0.0
    print(f"\nstride padding: type sizes {type_sizes} -> lcm {stride_lcm}, "
          f"x sector {SECTOR} -> pad to multiples of {pad_target:,}")
    if add == 0:
        print("  cost        : NONE - the natural stride already satisfies both "
              "constraints")
    else:
        print(f"  cost        : +{add:,} B on disk (+{pct:.3f}%)")
        print(f"  transfer    : +{pct:.3f}% bytes per expert miss")
        print("  est. decode : not estimated; measure on the target model, "
              "storage device, and arena configuration")
        print("  why         : ggml-cuda truncates nb[2]/type_size by integer "
              "division; an unpadded stride silently misaddresses every expert "
              "after the first")
    print(f"packed experts: {packed_total:,} B ({packed_total/1024**3:.2f} GiB)")
    print(f"other tensors : {other_total:,} B ({other_total/1024**3:.2f} GiB)")
    print(f"total         : {(packed_total+other_total)/1024**3:.2f} GiB "
          f"(source is {os.path.getsize(a.src)/1024**3:.2f} GiB)")
    if a.dry_run:
        return

    align = g.kv.get("general.alignment", 32)
    fin = open(a.src, "rb")

    # Tensor directory.
    #
    # Every layer gets one exact-size opaque byte tensor containing an
    # expert-major region:
    #
    #   region = [e0: gate|up|down][e1: gate|up|down] ...
    #
    # GGUF requires sequential tensor offsets, so overlapping strided directory
    # entries for gate/up/down are not expressible. The original projection
    # entries are therefore replaced by blk.L.ffn_exps_packed.weight. Metadata
    # below carries the part names, types, shapes, byte lengths, and stride; the
    # patched loader removes the opaque entry from its weight map and synthesises
    # the original logical projection tensors as strided views into this region.
    #
    # CONSEQUENCE: those views rely on mmap. The loader refuses expert-major
    # files when mmap is disabled because an ordinary contiguous copy of a
    # strided logical projection would read the wrong bytes.
    out_tensors = []
    for t in others:
        out_tensors.append((t.name, list(t.dims), t.type_id, t.nbytes, ("copy", t)))
    for i, L in enumerate(layers):
        nbytes = expert_bytes[i] * n_experts
        # 1-D I8 so the declared size is exact regardless of the quantisations
        # inside. GGUF REQUIRES sequential tensor offsets (gguf.cpp: "tensor
        # '%s' has offset %llu, expected %zu"), so overlapping strided entries
        # for gate/up/down are not expressible -- the loader reconstructs them
        # from the metadata below instead.
        out_tensors.append((f"blk.{L}.ffn_exps_packed.weight", [nbytes], 24,
                            nbytes, ("pack", L)))

    # Metadata is round-tripped with the `gguf` package rather than the local
    # reader. gguf_reader deliberately SKIPS long arrays (it stores
    # "<N skipped>"), including the tokenizer vocab and merges -- writing from
    # it would give a model that loads and then tokenises wrongly.
    from gguf import GGUFReader, GGUFWriter          # noqa: E402
    import numpy as np                                # noqa: E402

    rd = GGUFReader(a.src)
    arch = None
    for key, fld in rd.fields.items():
        if key == "general.architecture":
            arch = str(bytes(fld.parts[fld.data[0]]), "utf-8")
    wr = GGUFWriter(a.dst, arch or "deepseek4", use_temp_file=False)
    # 512 = volume logical sector. GGUF requires sequential tensor offsets, so
    # declaring this alignment is what makes every packed region start on a
    # sector boundary -- which direct I/O requires.
    wr.add_key_value("general.alignment", 512, 4)

    copied = 0
    for key, fld in rd.fields.items():
        if key.startswith("siliangem."):
            continue                                  # ours, added below
        if key == "general.alignment":
            continue                                  # we set our own (512)
        if key.startswith("GGUF."):
            # GGUF.version / tensor_count / kv_count are HEADER state that
            # GGUFReader surfaces as pseudo-fields; copying them as real KV
            # writes them twice and the file no longer parses.
            continue
        try:
            wr.add_key_value(key, fld.contents(), fld.types[0],
                             sub_type=fld.types[-1] if len(fld.types) > 1 else None)
            copied += 1
        except Exception as e:
            raise SystemExit(f"failed to copy metadata key {key!r}: {e}\n"
                             "Refusing to emit a GGUF with incomplete metadata.")
    print(f"metadata      : {copied} keys copied verbatim")

    # Everything the loader needs to rebuild gate/up/down from the opaque
    # packed tensor: part names, per-part sizes, quantisation types and shapes.
    part_types = [geom[L][p][3] for L in layers for p in parts]
    part_ne0   = [g.tensors[geom[L][p][2]].dims[0] for L in layers for p in parts]
    part_ne1   = [g.tensors[geom[L][p][2]].dims[1] for L in layers for p in parts]

    wr.add_key_value("siliangem.expert_major", True, 7)
    wr.add_key_value("siliangem.n_experts", int(n_experts), 4)
    wr.add_key_value("siliangem.part_names", ",".join(parts), 8)
    # per layer: byte stride between experts inside the packed tensor -> nb[2]
    # u32 not u64: llama.cpp get_arr accepts string/f32/u32/i32 arrays only.
    wr.add_key_value("siliangem.expert_bytes", [int(x) for x in expert_bytes], 9, sub_type=4)
    # layer-major, one entry per (layer, part): everything needed to rebuild
    # the logical tensor, since the packed tensor is opaque I8 bytes.
    wr.add_key_value("siliangem.part_bytes", [int(x) for x in part_bytes], 9, sub_type=4)
    wr.add_key_value("siliangem.part_types", [int(x) for x in part_types], 9, sub_type=4)
    wr.add_key_value("siliangem.part_ne0",   [int(x) for x in part_ne0],   9, sub_type=4)
    wr.add_key_value("siliangem.part_ne1",   [int(x) for x in part_ne1],   9, sub_type=4)
    # actual layer indices: do NOT assume 0..n-1
    wr.add_key_value("siliangem.layer_index", [int(L) for L in layers], 9, sub_type=4)

    from gguf import GGMLQuantizationType             # noqa: E402
    for name, dims, type_id, nbytes, _src in out_tensors:
        # raw_dtype must be the enum, not the integer id.
        # tensor_dtype must NOT be uint8: gguf then treats tensor_shape as a
        # BYTE shape and converts it to an element shape via the quant block
        # size. Our dims are already element counts, so that conversion would
        # corrupt them (and fails outright when bytes-per-row is not a multiple
        # of the type size). Any non-uint8 dtype keeps the shape verbatim;
        # raw_dtype is what actually gets stored.
        wr.add_tensor_info(name, list(reversed(dims)), np.dtype(np.float32),
                           nbytes, raw_dtype=GGMLQuantizationType(type_id))

    # ---- offsets: sequential, GGUF's only legal layout --------------------
    # gguf.cpp validates ti.offset == running sum of GGML_PAD(nbytes, alignment)
    # and rejects anything else. Declaring general.alignment = 512 therefore
    # gives every packed region a sector-aligned start for free, via the same
    # sequential rule -- no custom offsets needed.
    io_align = max(align, SECTOR)
    def roundup(x, a): return ((x + a - 1) // a) * a

    offsets = {}
    cur = 0
    for name, dims, type_id, nbytes, src in out_tensors:
        offsets[name] = cur
        cur += roundup(nbytes, io_align)
    data_total = cur

    wr.write_header_to_file()
    wr.write_kv_data_to_file()

    fout = wr.fout[0] if isinstance(wr.fout, list) else wr.fout
    ti = bytearray()
    for name, dims, type_id, nbytes, src in out_tensors:
        nb = name.encode("utf-8")
        ti += struct.pack("<Q", len(nb)) + nb
        ti += struct.pack("<I", len(dims))
        for d in dims:
            ti += struct.pack("<Q", d)
        ti += struct.pack("<I", type_id)
        ti += struct.pack("<Q", offsets[name])
    fout.write(ti)
    r = fout.tell() % io_align
    if r:
        fout.write(b"\0" * (io_align - r))
    data_start = fout.tell()
    print(f"data start    : {data_start:,}  (data section {data_total:,} B, "
          f"alignment {io_align})")

    written = 0

    for name, dims, type_id, nbytes, src in out_tensors:
        fout.seek(data_start + offsets[name])
        if src[0] == "copy":
            t = src[1]
            fin.seek(t.abs_offset)
            left = t.nbytes
            while left:
                chunk = fin.read(min(1 << 24, left))
                if not chunk:
                    raise SystemExit(f"short read on {name}")
                fout.write(chunk)
                left -= len(chunk)
        else:
            L = src[1]
            idx = layers.index(L)
            bases = {}
            for p in parts:
                per, _, tname, _tid = geom[L][p]
                bases[p] = (g.tensors[tname].abs_offset, per)
            pad_n = expert_bytes[idx] - raw_expert_bytes[idx]
            pad_buf = b"\0" * pad_n
            for e in range(n_experts):
                for p in parts:
                    base, per = bases[p]
                    fin.seek(base + e * per)
                    left = per
                    while left:
                        chunk = fin.read(min(1 << 24, left))
                        if not chunk:
                            raise SystemExit(f"short read on {name} expert {e}")
                        fout.write(chunk)
                        left -= len(chunk)
                if pad_n:
                    fout.write(pad_buf)   # never read by compute; alignment only
        written += nbytes
        if written % (1 << 30) < nbytes:
            print(f"  ... {written/1024**3:.1f} GiB", flush=True)

    # The last layer may end before data_total if its region is the final one;
    # make the file exactly as long as the directory claims.
    fout.seek(data_start + data_total - 1)
    fout.write(b"\0")
    fout.flush()
    wr.close()
    fin.close()
    print(f"done: {written/1024**3:.2f} GiB written to {a.dst}")
    print("NOTE: load this format with the bundled Siliang Engine build; "
          "unpatched llama.cpp does not understand the packed metadata. "
          "Run a separate --verify and the structural header probe before use.")


if __name__ == "__main__":
    main()
