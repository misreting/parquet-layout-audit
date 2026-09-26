from __future__ import annotations

import pytest

from parquet_layout_audit.thrift_compact import BINARY, I32, LIST, Reader, ThriftDecodeError


def test_varint_roundtrip_small_and_multibyte() -> None:
    # 300 needs two LEB128 bytes: 0b1_0101100 -> [0xAC, 0x02]
    assert Reader(bytes([0xAC, 0x02])).read_varint() == 300
    assert Reader(bytes([0x00])).read_varint() == 0
    assert Reader(bytes([0x7F])).read_varint() == 127


def test_zigzag_decoding_covers_negative_and_positive() -> None:
    # zigzag(0) = 0, zigzag(-1) = 1, zigzag(1) = 2, zigzag(-2) = 3
    assert Reader(bytes([0])).read_zigzag() == 0
    assert Reader(bytes([1])).read_zigzag() == -1
    assert Reader(bytes([2])).read_zigzag() == 1
    assert Reader(bytes([3])).read_zigzag() == -2


def test_binary_and_string_read_length_prefixed_bytes() -> None:
    data = bytes([5]) + b"hello"
    assert Reader(data).read_binary() == b"hello"
    assert Reader(data).read_string() == "hello"


def test_field_header_short_form_uses_delta_from_previous_field() -> None:
    header_byte = (1 << 4) | I32
    r = Reader(bytes([header_byte, 0x00]))
    field = r.read_field_begin()
    assert field is not None
    assert field.id == 1
    assert field.type == I32
    assert r.read_zigzag() == 0

    header_byte2 = (2 << 4) | I32
    r2 = Reader(bytes([header_byte, 0x00, header_byte2, 0x00]))
    f1 = r2.read_field_begin()
    assert f1 is not None and f1.id == 1
    r2.read_zigzag()
    f2 = r2.read_field_begin()
    assert f2 is not None and f2.id == 3


def test_field_header_long_form_reads_explicit_zigzag_field_id() -> None:
    header_byte = (0 << 4) | I32
    r = Reader(bytes([header_byte, 0x28]))  # zigzag(20) = 40 = 0x28
    field = r.read_field_begin()
    assert field is not None
    assert field.id == 20
    assert field.type == I32


def test_boolean_field_value_is_folded_into_the_header_nibble() -> None:
    true_header = (1 << 4) | 0x01
    false_header = (2 << 4) | 0x02
    r = Reader(bytes([true_header, false_header]))
    f1 = r.read_field_begin()
    assert f1 is not None
    assert f1.bool_value is True
    f2 = r.read_field_begin()
    assert f2 is not None
    assert f2.bool_value is False


def test_stop_byte_ends_a_struct() -> None:
    r = Reader(bytes([0x00]))
    assert r.read_field_begin() is None


def test_list_header_short_and_long_form_sizes() -> None:
    header = (3 << 4) | BINARY
    etype, size = Reader(bytes([header])).read_collection_begin()
    assert (etype, size) == (BINARY, 3)

    header_long = (15 << 4) | BINARY
    etype2, size2 = Reader(bytes([header_long, 0x14])).read_collection_begin()
    assert (etype2, size2) == (BINARY, 20)


def test_skip_advances_past_nested_struct_and_list_without_interpreting_values() -> None:
    inner_list_header = (2 << 4) | I32
    payload = bytes(
        [
            (1 << 4) | LIST,  # field 1, type LIST
            inner_list_header,
            20,
            20,
            0x00,  # STOP
        ]
    )
    r = Reader(payload)
    field = r.read_field_begin()
    assert field is not None
    r.skip(field.type)
    assert r.read_field_begin() is None


def test_unknown_type_id_raises_decode_error() -> None:
    r = Reader(b"\x00")
    with pytest.raises(ThriftDecodeError):
        r.skip(0xFF)


def test_reading_past_end_of_buffer_raises_decode_error() -> None:
    r = Reader(b"")
    with pytest.raises(ThriftDecodeError):
        r.read_byte()
