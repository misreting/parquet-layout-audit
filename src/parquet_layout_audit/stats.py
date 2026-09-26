"""Decode raw Parquet ``Statistics`` min/max bytes into comparable values.

Parquet stores column-chunk statistics as raw bytes, using an encoding that
depends on the column's physical type (and, for a couple of cases, its
logical type). To estimate prunability we need to compare min/max values
across row groups with ``<=``/``>=``, so every supported type is decoded
into a plain Python ``int``, ``float``, ``bool``, or ``bytes`` -- all of
which compare the way Parquet's own sort order defines for that type.

What is *not* supported is documented in :mod:`parquet_layout_audit.analyze`
and the README rather than guessed at here: ``INT96`` (deprecated, no
reliable interpretation without extra convention) and any physical type this
module doesn't recognise decode to ``None``, which callers treat the same as
"no statistics" for that column.
"""

from __future__ import annotations

import struct

# Converted/logical-type spellings (as produced by
# parquet_layout_audit.footer) that mean "this INT32/INT64 is unsigned".
_UNSIGNED_CONVERTED = {"UINT_8", "UINT_16", "UINT_32", "UINT_64"}


def decode_stat_value(
    raw: bytes | None,
    *,
    physical_type: str,
    converted_type: str | None,
    logical_type: str | None,
) -> object | None:
    """Decode one min/max statistics value. Returns ``None`` if undecodable.

    ``raw`` itself being ``None`` (the writer didn't set this bound) also
    returns ``None`` -- callers don't need to distinguish "absent" from
    "unsupported type" since both mean the same thing for pruning purposes.
    """
    if raw is None:
        return None

    if physical_type == "BOOLEAN":
        if len(raw) != 1:
            return None
        return raw[0] != 0

    if physical_type == "INT32":
        if len(raw) != 4:
            return None
        fmt = "<I" if converted_type in _UNSIGNED_CONVERTED else "<i"
        return int(struct.unpack(fmt, raw)[0])

    if physical_type == "INT64":
        if len(raw) != 8:
            return None
        fmt = "<Q" if converted_type in _UNSIGNED_CONVERTED else "<q"
        return int(struct.unpack(fmt, raw)[0])

    if physical_type == "FLOAT":
        if len(raw) != 4:
            return None
        return float(struct.unpack("<f", raw)[0])

    if physical_type == "DOUBLE":
        if len(raw) != 8:
            return None
        return float(struct.unpack("<d", raw)[0])

    if physical_type == "BYTE_ARRAY":
        # Parquet's sort order for BYTE_ARRAY/STRING is unsigned byte-wise
        # comparison, which is exactly what Python's `bytes` ordering does.
        return raw

    if physical_type == "FIXED_LEN_BYTE_ARRAY":
        if logical_type is not None and logical_type.startswith("DECIMAL"):
            # Decimal-on-fixed-length-bytes is a big-endian two's-complement
            # integer; comparing the integers preserves decimal order
            # regardless of scale, since all values in one column share it.
            return int.from_bytes(raw, byteorder="big", signed=True)
        # No decimal annotation: most likely a UUID or other opaque fixed
        # blob. Byte-wise comparison isn't guaranteed meaningful for these,
        # but it's the same fallback Parquet's own writers use to compute
        # min/max for such columns, so it's what we compare too.
        return raw

    # INT96 (deprecated) and anything unrecognised: no decode.
    return None


def is_orderable(physical_type: str) -> bool:
    """Whether this tool can decode statistics for the given physical type.

    Used to distinguish "this column has no statistics because the writer
    didn't record them" from "this column's type isn't one we can compare"
    (currently only ``INT96``) -- the two get different remedies.
    """
    return physical_type != "INT96"
