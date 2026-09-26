"""Read a Parquet file's footer and decode it into a :class:`FileLayout`.

A Parquet file ends with::

    <file metadata (Thrift compact-encoded FileMetaData)>
    <4-byte little-endian footer length>
    <4-byte magic: b"PAR1">

So the whole file's data never has to be touched: seek to the last 8 bytes,
read the length, seek back that far, and decode. This follows the same
approach as the sibling project ``parquet-schema-check`` (see its
``footer.py``) against ``FileMetaData`` (see ``parquet.thrift`` in the
Apache Parquet format repository) using the compact-protocol primitives in
:mod:`parquet_layout_audit.thrift_compact`.

This module reads more of ``FileMetaData`` than a schema check needs: row
group byte sizes and row counts, each column chunk's ``num_values``,
compressed/uncompressed sizes, dictionary-page presence, and -- the whole
point of this tool -- the ``Statistics`` struct's min/max bounds per column
chunk, needed to estimate row-group pruning. Page counts come from the
optional ``encoding_stats`` field (a per-column-chunk tally of how many
pages of each type the writer emitted); when a writer doesn't record it,
this tool reports the page count as unknown rather than estimating it by
scanning page headers (which would mean reading every row group's raw
bytes, defeating the point of a footer-only tool).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from .stats import decode_stat_value
from .thrift_compact import Reader, ThriftDecodeError
from .types import ColumnChunkLayout, ColumnStatistics, FileLayout, RowGroupLayout

MAGIC = b"PAR1"
ENCRYPTED_MAGIC = b"PARE"
FOOTER_TRAILER_SIZE = 8  # 4-byte length + 4-byte magic

PHYSICAL_TYPE = {
    0: "BOOLEAN",
    1: "INT32",
    2: "INT64",
    3: "INT96",
    4: "FLOAT",
    5: "DOUBLE",
    6: "BYTE_ARRAY",
    7: "FIXED_LEN_BYTE_ARRAY",
}

CONVERTED_TYPE = {
    0: "UTF8",
    1: "MAP",
    2: "MAP_KEY_VALUE",
    3: "LIST",
    4: "ENUM",
    5: "DECIMAL",
    6: "DATE",
    7: "TIME_MILLIS",
    8: "TIME_MICROS",
    9: "TIMESTAMP_MILLIS",
    10: "TIMESTAMP_MICROS",
    11: "UINT_8",
    12: "UINT_16",
    13: "UINT_32",
    14: "UINT_64",
    15: "INT_8",
    16: "INT_16",
    17: "INT_32",
    18: "INT_64",
    19: "JSON",
    20: "BSON",
    21: "INTERVAL",
}

ENCODING = {
    0: "PLAIN",
    2: "PLAIN_DICTIONARY",
    3: "RLE",
    4: "BIT_PACKED",
    5: "DELTA_BINARY_PACKED",
    6: "DELTA_LENGTH_BYTE_ARRAY",
    7: "DELTA_BYTE_ARRAY",
    8: "RLE_DICTIONARY",
    9: "BYTE_STREAM_SPLIT",
}

DICTIONARY_ENCODINGS = {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}

COMPRESSION_CODEC = {
    0: "UNCOMPRESSED",
    1: "SNAPPY",
    2: "GZIP",
    3: "LZO",
    4: "BROTLI",
    5: "LZ4",
    6: "ZSTD",
    7: "LZ4_RAW",
}

# PageType enum (used inside PageEncodingStats).
_DATA_PAGE = 0
_INDEX_PAGE = 1
_DICTIONARY_PAGE = 2
_DATA_PAGE_V2 = 3

_TIME_UNIT = {1: "ms", 2: "us", 3: "ns"}

# LogicalType union field ids that carry no payload of their own.
_LOGICAL_EMPTY = {
    1: "STRING",
    2: "MAP",
    3: "LIST",
    4: "ENUM",
    6: "DATE",
    11: "NULL",
    12: "JSON",
    13: "BSON",
    14: "UUID",
    15: "FLOAT16",
}


class ParquetFooterError(ValueError):
    """The file is not a readable, unencrypted Parquet footer."""


@dataclass
class _SchemaElementRaw:
    """One decoded ``SchemaElement`` before it's placed in the schema tree."""

    type: str | None = None
    type_length: int | None = None
    repetition_type: str | None = None
    name: str = ""
    num_children: int | None = None
    converted_type: str | None = None
    scale: int | None = None
    precision: int | None = None
    logical_type: str | None = None


@dataclass
class _LeafColumn:
    element: _SchemaElementRaw
    path: tuple[str, ...]


@dataclass
class _StatisticsRaw:
    max_legacy: bytes | None = None
    min_legacy: bytes | None = None
    null_count: int | None = None
    distinct_count: int | None = None
    max_value: bytes | None = None
    min_value: bytes | None = None
    is_max_exact: bool = True
    is_min_exact: bool = True

    @property
    def has_any_bound(self) -> bool:
        return self.max_value is not None or self.min_value is not None or self.max_legacy is not None or self.min_legacy is not None


@dataclass
class _ColumnMetaRaw:
    physical_type: str | None = None
    encodings: set[str] = field(default_factory=set)
    path_in_schema: tuple[str, ...] | None = None
    codec: str | None = None
    num_values: int = 0
    total_uncompressed_size: int = 0
    total_compressed_size: int = 0
    dictionary_page_offset: int | None = None
    statistics: _StatisticsRaw | None = None
    data_page_count: int | None = None  # from encoding_stats, if the writer recorded it


@dataclass
class _RowGroupRaw:
    columns: list[_ColumnMetaRaw] = field(default_factory=list)
    total_byte_size: int = 0
    num_rows: int = 0
    total_compressed_size: int | None = None
    sorting_columns: list[tuple[int, bool]] = field(default_factory=list)


@dataclass
class _FileMetadataRaw:
    schema: list[_SchemaElementRaw] = field(default_factory=list)
    num_rows: int = 0
    created_by: str | None = None
    row_groups: list[_RowGroupRaw] = field(default_factory=list)


def _read_logical_type(r: Reader) -> str | None:
    """Decode a LogicalType union into a canonical descriptive string."""
    r.enter_struct()
    result: str | None = None
    try:
        while True:
            f = r.read_field_begin()
            if f is None:
                break
            if f.id in _LOGICAL_EMPTY:
                result = _LOGICAL_EMPTY[f.id]
                r.skip(f.type)  # empty struct body, but still must consume STOP
            elif f.id == 5:  # DecimalType { scale, precision }
                r.enter_struct()
                scale = precision = None
                while True:
                    sf = r.read_field_begin()
                    if sf is None:
                        break
                    if sf.id == 1:
                        scale = r.read_zigzag()
                    elif sf.id == 2:
                        precision = r.read_zigzag()
                    else:
                        r.skip(sf.type)
                r.exit_struct()
                result = f"DECIMAL(precision={precision}, scale={scale})"
            elif f.id in (7, 8):  # TimeType / TimestampType { isAdjustedToUTC, unit }
                kind = "TIME" if f.id == 7 else "TIMESTAMP"
                r.enter_struct()
                utc = None
                unit = None
                while True:
                    sf = r.read_field_begin()
                    if sf is None:
                        break
                    if sf.id == 1:
                        utc = sf.bool_value
                    elif sf.id == 2:  # TimeUnit union
                        r.enter_struct()
                        while True:
                            uf = r.read_field_begin()
                            if uf is None:
                                break
                            if uf.id in _TIME_UNIT:
                                unit = _TIME_UNIT[uf.id]
                            r.skip(uf.type)
                        r.exit_struct()
                    else:
                        r.skip(sf.type)
                r.exit_struct()
                tz = "utc" if utc else "local"
                result = f"{kind}(unit={unit}, {tz})"
            elif f.id == 10:  # IntType { bitWidth, isSigned }
                r.enter_struct()
                bit_width = None
                signed = None
                while True:
                    sf = r.read_field_begin()
                    if sf is None:
                        break
                    if sf.id == 1:
                        bit_width = r.read_varint()
                    elif sf.id == 2:
                        signed = sf.bool_value
                    else:
                        r.skip(sf.type)
                r.exit_struct()
                result = f"INT(bitWidth={bit_width}, signed={signed})"
            else:
                r.skip(f.type)
    finally:
        r.exit_struct()
    return result


def _read_schema_element(r: Reader) -> _SchemaElementRaw:
    element = _SchemaElementRaw()
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 1:
            element.type = PHYSICAL_TYPE.get(r.read_zigzag())
        elif f.id == 2:
            element.type_length = r.read_zigzag()
        elif f.id == 3:
            element.repetition_type = {0: "REQUIRED", 1: "OPTIONAL", 2: "REPEATED"}.get(r.read_zigzag())
        elif f.id == 4:
            element.name = r.read_string()
        elif f.id == 5:
            element.num_children = r.read_zigzag()
        elif f.id == 6:
            element.converted_type = CONVERTED_TYPE.get(r.read_zigzag())
        elif f.id == 7:
            element.scale = r.read_zigzag()
        elif f.id == 8:
            element.precision = r.read_zigzag()
        elif f.id == 10:
            element.logical_type = _read_logical_type(r)
        else:
            r.skip(f.type)
    r.exit_struct()
    return element


def _read_statistics(r: Reader) -> _StatisticsRaw:
    stats = _StatisticsRaw()
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 1:
            stats.max_legacy = r.read_binary()
        elif f.id == 2:
            stats.min_legacy = r.read_binary()
        elif f.id == 3:
            stats.null_count = r.read_zigzag()
        elif f.id == 4:
            stats.distinct_count = r.read_zigzag()
        elif f.id == 5:
            stats.max_value = r.read_binary()
        elif f.id == 6:
            stats.min_value = r.read_binary()
        elif f.id == 7:
            stats.is_max_exact = bool(f.bool_value) if f.bool_value is not None else True
        elif f.id == 8:
            stats.is_min_exact = bool(f.bool_value) if f.bool_value is not None else True
        else:
            r.skip(f.type)
    r.exit_struct()
    return stats


def _read_page_encoding_stats(r: Reader, size: int) -> int:
    """Read a ``list<PageEncodingStats>`` and return the total DATA_PAGE(_V2) count."""
    total = 0
    for _ in range(size):
        r.enter_struct()
        page_type = None
        count = 0
        while True:
            f = r.read_field_begin()
            if f is None:
                break
            if f.id == 1:
                page_type = r.read_zigzag()
            elif f.id == 3:
                count = r.read_zigzag()
            else:
                r.skip(f.type)
        r.exit_struct()
        if page_type in (_DATA_PAGE, _DATA_PAGE_V2):
            total += count
    return total


def _read_column_meta_data(r: Reader) -> _ColumnMetaRaw:
    meta = _ColumnMetaRaw()
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 1:
            meta.physical_type = PHYSICAL_TYPE.get(r.read_zigzag())
        elif f.id == 2:  # list<Encoding>
            _etype, size = r.read_collection_begin()
            for _ in range(size):
                meta.encodings.add(ENCODING.get(r.read_zigzag(), "UNKNOWN"))
        elif f.id == 3:  # list<string> path_in_schema
            _etype, size = r.read_collection_begin()
            meta.path_in_schema = tuple(r.read_string() for _ in range(size))
        elif f.id == 4:
            meta.codec = COMPRESSION_CODEC.get(r.read_zigzag(), "UNKNOWN")
        elif f.id == 5:
            meta.num_values = r.read_zigzag()
        elif f.id == 6:
            meta.total_uncompressed_size = r.read_zigzag()
        elif f.id == 7:
            meta.total_compressed_size = r.read_zigzag()
        elif f.id == 11:
            meta.dictionary_page_offset = r.read_zigzag()
        elif f.id == 12:
            meta.statistics = _read_statistics(r)
        elif f.id == 13:  # list<PageEncodingStats>
            _etype, size = r.read_collection_begin()
            meta.data_page_count = _read_page_encoding_stats(r, size)
        else:
            r.skip(f.type)
    r.exit_struct()
    return meta


def _read_column_chunk(r: Reader) -> _ColumnMetaRaw | None:
    meta_data: _ColumnMetaRaw | None = None
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 3:
            meta_data = _read_column_meta_data(r)
        else:
            r.skip(f.type)
    r.exit_struct()
    return meta_data


def _read_sorting_column(r: Reader) -> tuple[int, bool]:
    r.enter_struct()
    column_idx = 0
    descending = False
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 1:
            column_idx = r.read_zigzag()
        elif f.id == 2:
            descending = bool(f.bool_value)
        else:
            r.skip(f.type)
    r.exit_struct()
    return column_idx, descending


def _read_row_group(r: Reader) -> _RowGroupRaw:
    rg = _RowGroupRaw()
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 1:  # list<ColumnChunk>
            _etype, size = r.read_collection_begin()
            for _ in range(size):
                meta = _read_column_chunk(r)
                if meta is not None:
                    rg.columns.append(meta)
        elif f.id == 2:
            rg.total_byte_size = r.read_zigzag()
        elif f.id == 3:
            rg.num_rows = r.read_zigzag()
        elif f.id == 4:  # list<SortingColumn>
            _etype, size = r.read_collection_begin()
            rg.sorting_columns = [_read_sorting_column(r) for _ in range(size)]
        elif f.id == 6:
            rg.total_compressed_size = r.read_zigzag()
        else:
            r.skip(f.type)
    r.exit_struct()
    return rg


def _read_file_metadata(data: bytes) -> _FileMetadataRaw:
    r = Reader(data)
    result = _FileMetadataRaw()
    r.enter_struct()
    while True:
        f = r.read_field_begin()
        if f is None:
            break
        if f.id == 2:  # list<SchemaElement>
            _etype, size = r.read_collection_begin()
            result.schema = [_read_schema_element(r) for _ in range(size)]
        elif f.id == 3:
            result.num_rows = r.read_zigzag()
        elif f.id == 4:  # list<RowGroup>
            _etype, size = r.read_collection_begin()
            result.row_groups = [_read_row_group(r) for _ in range(size)]
        elif f.id == 6:
            result.created_by = r.read_string()
        else:
            r.skip(f.type)
    r.exit_struct()
    return result


def _build_leaves(elements: list[_SchemaElementRaw]) -> list[_LeafColumn]:
    """Walk the flattened, preorder SchemaElement list into leaf columns.

    Parquet stores the schema tree flattened: a root ``message`` element
    (no ``type``, only ``num_children``) followed by its descendants in
    preorder, each group again giving its own ``num_children``. A leaf is
    any element that *does* carry a physical ``type``.
    """
    if not elements:
        return []

    leaves: list[_LeafColumn] = []

    def walk(index: int, path: tuple[str, ...]) -> int:
        element = elements[index]
        this_path = path + (element.name,) if index != 0 else ()
        if element.type is not None:
            leaves.append(_LeafColumn(element=element, path=this_path))
            return index + 1
        next_index = index + 1
        for _ in range(element.num_children or 0):
            next_index = walk(next_index, this_path)
        return next_index

    walk(0, ())
    return leaves


def _canonical_logical_type(element: _SchemaElementRaw) -> str | None:
    if element.logical_type is not None:
        return element.logical_type
    converted = element.converted_type
    if converted is None:
        return None
    if converted == "DECIMAL":
        return f"DECIMAL(precision={element.precision}, scale={element.scale})"
    if converted == "TIMESTAMP_MILLIS":
        return "TIMESTAMP(unit=ms, local)"
    if converted == "TIMESTAMP_MICROS":
        return "TIMESTAMP(unit=us, local)"
    if converted == "TIME_MILLIS":
        return "TIME(unit=ms, local)"
    if converted == "TIME_MICROS":
        return "TIME(unit=us, local)"
    return converted


def _build_statistics(raw: _StatisticsRaw | None, element: _SchemaElementRaw) -> ColumnStatistics:
    if raw is None or not raw.has_any_bound:
        # Still surface null_count/distinct_count if the writer recorded
        # those without min/max (rare, but distinct from "no stats at all").
        if raw is not None and (raw.null_count is not None or raw.distinct_count is not None):
            return ColumnStatistics(
                has_stats=False,
                null_count=raw.null_count,
                distinct_count=raw.distinct_count,
            )
        return ColumnStatistics(has_stats=False)

    physical_type = element.type or ""
    logical_type = _canonical_logical_type(element)
    # Prefer the modern min_value/max_value; fall back to the deprecated
    # min/max fields when a writer only set those (pre-2.6 writers).
    min_raw = raw.min_value if raw.min_value is not None else raw.min_legacy
    max_raw = raw.max_value if raw.max_value is not None else raw.max_legacy

    min_decoded = decode_stat_value(
        min_raw, physical_type=physical_type, converted_type=element.converted_type, logical_type=logical_type
    )
    max_decoded = decode_stat_value(
        max_raw, physical_type=physical_type, converted_type=element.converted_type, logical_type=logical_type
    )
    has_stats = min_decoded is not None and max_decoded is not None
    return ColumnStatistics(
        has_stats=has_stats,
        min_value=min_decoded,
        max_value=max_decoded,
        null_count=raw.null_count,
        distinct_count=raw.distinct_count,
        is_min_exact=raw.is_min_exact,
        is_max_exact=raw.is_max_exact,
    )


def parse_footer_bytes(data: bytes, *, source: str, file_size: int) -> FileLayout:
    try:
        metadata = _read_file_metadata(data)
    except ThriftDecodeError as exc:
        raise ParquetFooterError(f"{source}: could not decode footer metadata: {exc}") from exc

    leaves = _build_leaves(metadata.schema)
    leaf_by_path = {leaf.path: leaf.element for leaf in leaves}

    row_groups = []
    for rg in metadata.row_groups:
        columns = []
        for col in rg.columns:
            path = col.path_in_schema or ()
            element = leaf_by_path.get(path)
            physical_type = (element.type if element is not None else None) or col.physical_type or "UNKNOWN"
            statistics = _build_statistics(col.statistics, element) if element is not None else ColumnStatistics(has_stats=False)
            columns.append(
                ColumnChunkLayout(
                    path=path,
                    physical_type=physical_type,
                    num_values=col.num_values,
                    total_uncompressed_size=col.total_uncompressed_size,
                    total_compressed_size=col.total_compressed_size,
                    encodings=frozenset(col.encodings),
                    codec=col.codec,
                    has_dictionary_page=col.dictionary_page_offset is not None,
                    statistics=statistics,
                    data_page_count=col.data_page_count,
                )
            )
        row_groups.append(
            RowGroupLayout(
                num_rows=rg.num_rows,
                total_byte_size=rg.total_byte_size,
                total_compressed_size=rg.total_compressed_size,
                sorting_columns=tuple(rg.sorting_columns),
                columns=tuple(columns),
            )
        )

    return FileLayout(
        path=source,
        file_size=file_size,
        num_rows=metadata.num_rows,
        created_by=metadata.created_by,
        row_groups=tuple(row_groups),
    )


def _read_footer_bytes(f: io.BufferedIOBase, file_size: int) -> bytes:
    if file_size < FOOTER_TRAILER_SIZE + len(MAGIC):
        raise ParquetFooterError("file is too small to be a Parquet file")

    f.seek(file_size - FOOTER_TRAILER_SIZE)
    trailer = f.read(FOOTER_TRAILER_SIZE)
    footer_length = int.from_bytes(trailer[0:4], "little", signed=False)
    magic = trailer[4:8]

    if magic == ENCRYPTED_MAGIC:
        raise ParquetFooterError(
            "file has an encrypted footer (magic 'PARE'); this tool cannot read"
            " encrypted Parquet metadata"
        )
    if magic != MAGIC:
        raise ParquetFooterError(f"not a Parquet file: expected magic {MAGIC!r}, found {magic!r}")

    footer_start = file_size - FOOTER_TRAILER_SIZE - footer_length
    if footer_start < len(MAGIC):
        raise ParquetFooterError("declared footer length is larger than the file")

    f.seek(footer_start)
    return f.read(footer_length)


def read_layout(path: str | Path) -> FileLayout:
    """Read a Parquet file's footer from disk and decode its row-group layout.

    Only the last ``footer_length + 8`` bytes are read -- the row data is
    never touched.
    """
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        footer_bytes = _read_footer_bytes(f, file_size)
    return parse_footer_bytes(footer_bytes, source=str(path), file_size=file_size)
