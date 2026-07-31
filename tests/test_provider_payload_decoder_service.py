import pytest

from app.services.provider_payload_decoder_service import (
    ProviderPayloadDecodeError,
    ProviderPayloadDecoderService,
)


def test_jsonl_decoder_accepts_object_records():
    rows = ProviderPayloadDecoderService().decode(
        b'{"region":"a","count":1}\n{"region":"b","count":2}\n',
        data_format="jsonl",
        max_rows=10,
    )

    assert rows == [
        {"region": "a", "count": 1},
        {"region": "b", "count": 2},
    ]


def test_csv_decoder_rejects_duplicate_columns():
    with pytest.raises(ProviderPayloadDecodeError) as exc_info:
        ProviderPayloadDecoderService().decode(
            b"region,region\na,b\n",
            data_format="csv",
            max_rows=10,
        )

    assert exc_info.value.reason_code == "duplicate_columns"


def test_decoder_enforces_configured_row_limit():
    with pytest.raises(ProviderPayloadDecodeError) as exc_info:
        ProviderPayloadDecoderService().decode(
            b'{"region":"a"}\n{"region":"b"}\n',
            data_format="jsonl",
            max_rows=1,
        )

    assert exc_info.value.reason_code == "row_limit_exceeded"
