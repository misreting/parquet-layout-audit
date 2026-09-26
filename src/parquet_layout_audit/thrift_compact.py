"""A minimal reader for Apache Thrift's compact protocol.

Parquet footers are Thrift-compact-encoded ``FileMetaData`` structs. This
module implements just enough of the compact protocol to walk that binary
format: varints, zigzag signed integers, the field-header delta encoding,
and the container (list/set/map) headers. It has no notion of a ``.thrift``
IDL or code generation -- callers read fields by id and skip what they
don't need.

This mirrors the approach in the sibling project ``parquet-schema-check``
(see its ``thrift_compact.py``), which established that hand-rolling this
is tractable and avoids a third-party Thrift or Parquet dependency. This
copy is kept in this package rather than imported from that one because
neither package depends on the other -- both are meant to be installed and
used standalone. It is not a general Thrift library: there is no support
for the binary protocol, no framed transport, and no write path.
"""

from __future__ import annotations

from dataclasses import dataclass

# Compact protocol type ids (as they appear in field headers and struct
# contexts). Collection headers reuse a distinct short table but boolean
# is the only value that differs (see COLLECTION_BOOL below).
STOP = 0x00
BOOLEAN_TRUE = 0x01
BOOLEAN_FALSE = 0x02
BYTE = 0x03
I16 = 0x04
I32 = 0x05
I64 = 0x06
DOUBLE = 0x07
BINARY = 0x08
LIST = 0x09
SET = 0x0A
MAP = 0x0B
STRUCT = 0x0C

# Inside a list/set/map, booleans are not folded into the header the way
# they are for struct fields -- there is no field header to fold them into.
# The element type recorded in the collection header is still the BOOLEAN_TRUE
# marker (0x01); each element is then read as one full byte, 0x01 or 0x00.
COLLECTION_BOOL = BOOLEAN_TRUE


class ThriftDecodeError(ValueError):
    """The byte stream is not a well-formed compact-protocol struct."""


def _zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


@dataclass
class Field:
    """One decoded struct field: its Thrift field id and raw type id.

    ``bool_value`` is set (to ``True``/``False``) only when this field's
    type nibble was the compact-protocol boolean shortcut, in which case
    the value is already fully decoded and there is nothing left to read.
    """

    id: int
    type: int
    bool_value: bool | None = None


class Reader:
    """A cursor over compact-protocol bytes.

    Tracks the "last field id" per struct nesting level, since compact
    protocol field ids are delta-encoded against the previous sibling
    field within the *same* struct.
    """

    __slots__ = ("data", "pos", "_field_stack")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos
        self._field_stack: list[int] = [0]

    # -- primitives ---------------------------------------------------

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise ThriftDecodeError("unexpected end of buffer reading a byte")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def read_bytes(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0 or end > len(self.data):
            raise ThriftDecodeError(f"unexpected end of buffer reading {n} bytes")
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def read_varint(self) -> int:
        """Unsigned LEB128. Used for lengths, sizes, and zigzag payloads."""
        result = 0
        shift = 0
        while True:
            b = self.read_byte()
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7
            if shift > 70:
                raise ThriftDecodeError("varint too long")

    def read_zigzag(self) -> int:
        return _zigzag_decode(self.read_varint())

    def read_double(self) -> float:
        import struct

        (value,) = struct.unpack("<d", self.read_bytes(8))
        return float(value)

    def read_binary(self) -> bytes:
        length = self.read_varint()
        return self.read_bytes(length)

    def read_string(self) -> str:
        return self.read_binary().decode("utf-8", errors="replace")

    # -- struct / field headers ----------------------------------------

    def enter_struct(self) -> None:
        self._field_stack.append(0)

    def exit_struct(self) -> None:
        self._field_stack.pop()

    def read_field_begin(self) -> Field | None:
        """Read one field header. Returns None at STOP (end of struct)."""
        header = self.read_byte()
        if header == STOP:
            return None
        modifier = (header >> 4) & 0x0F
        ttype = header & 0x0F
        if modifier == 0:
            field_id = self.read_zigzag()
        else:
            field_id = self._field_stack[-1] + modifier
        self._field_stack[-1] = field_id

        if ttype in (BOOLEAN_TRUE, BOOLEAN_FALSE):
            return Field(id=field_id, type=ttype, bool_value=(ttype == BOOLEAN_TRUE))
        return Field(id=field_id, type=ttype)

    def read_collection_begin(self) -> tuple[int, int]:
        """Read a list/set header. Returns (element_type, size)."""
        header = self.read_byte()
        size = (header >> 4) & 0x0F
        etype = header & 0x0F
        if size == 15:
            size = self.read_varint()
        return etype, size

    def read_map_begin(self) -> tuple[int, int, int]:
        """Read a map header. Returns (key_type, value_type, size)."""
        size = self.read_varint()
        if size == 0:
            return (STOP, STOP, 0)
        header = self.read_byte()
        ktype = (header >> 4) & 0x0F
        vtype = header & 0x0F
        return ktype, vtype, size

    def read_bool_element(self) -> bool:
        return self.read_byte() == 0x01

    # -- generic skip ---------------------------------------------------

    def skip(self, ttype: int) -> None:
        """Advance past one value of the given type without interpreting it.

        Used for every field this module doesn't need, so unrecognised or
        forward-compatible struct fields don't break decoding.
        """
        if ttype in (BOOLEAN_TRUE, BOOLEAN_FALSE):
            return
        if ttype == BYTE:
            self.read_byte()
            return
        if ttype in (I16, I32, I64):
            self.read_varint()
            return
        if ttype == DOUBLE:
            self.read_bytes(8)
            return
        if ttype == BINARY:
            self.read_binary()
            return
        if ttype in (LIST, SET):
            etype, size = self.read_collection_begin()
            for _ in range(size):
                if etype == COLLECTION_BOOL:
                    self.read_bool_element()
                else:
                    self.skip(etype)
            return
        if ttype == MAP:
            ktype, vtype, size = self.read_map_begin()
            for _ in range(size):
                if ktype == COLLECTION_BOOL:
                    self.read_bool_element()
                else:
                    self.skip(ktype)
                if vtype == COLLECTION_BOOL:
                    self.read_bool_element()
                else:
                    self.skip(vtype)
            return
        if ttype == STRUCT:
            self.enter_struct()
            while True:
                field = self.read_field_begin()
                if field is None:
                    break
                self.skip(field.type)
            self.exit_struct()
            return
        raise ThriftDecodeError(f"unknown compact-protocol type id {ttype!r}")
