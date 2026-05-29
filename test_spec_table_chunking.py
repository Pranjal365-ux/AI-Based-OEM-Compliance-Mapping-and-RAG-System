import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ingestion"))

from chunker import chunk_pages
from meta_extractor import _extract_models


def _doc_meta(models=None):
    return {
        "vendor": "Fortinet",
        "family": "FortiGate 7000F Series",
        "product_family": "FortiGate 7000F Series",
        "display_family": "FortiGate 7000F Series",
        "models": models or ["FG-7081F", "FG-7081F-DC", "FG-7121F"],
        "category": "NGFW",
        "doc_name": "unit-test.pdf",
    }


def test_crypto_and_family_ids_are_not_models():
    text = """
    FortiGate 7000F Series
    SHA-256 SHA-384 SHA-512
    FG-7000F Series
    FG-7081F FG-7081F-DC FG-7121F
    PA-3200 Series PA-3220 PA-3250 PA-3260
    """

    models = _extract_models(text, "FortiGate 7000F Series")

    assert "SHA-256" not in models
    assert "SHA-384" not in models
    assert "SHA-512" not in models
    assert "FG-7000F" not in models
    assert "PA-3200" not in models
    assert "FG-7081F" in models
    assert "FG-7081F-DC" in models
    assert "PA-3220" in models


def test_model_columns_create_one_chunk_per_spec_cell():
    pages = [
        {
            "page": 1,
            "text": """
[TABLE]
Specification | FG-7081F | FG-7081F-DC | FG-7121F
--- | --- | --- | ---
Firewall Throughput | 500 Gbps | 500 Gbps | 700 Gbps
IPS Throughput | 300 Gbps | 300 Gbps | 450 Gbps
Concurrent Sessions | 100M | 100M | 150M
[/TABLE]
""",
        }
    ]

    chunks = chunk_pages(pages, _doc_meta())
    spec_rows = [c for c in chunks if c["metadata"]["chunk_type"] == "spec_row"]

    assert len(spec_rows) == 9
    target = [
        c
        for c in spec_rows
        if c["metadata"]["model"] == "FG-7121F"
        and c["metadata"]["spec_name"] == "IPS Throughput"
    ][0]
    assert target["metadata"]["spec_value"] == "450 Gbps"


def test_generic_headers_do_not_become_models():
    pages = [
        {
            "page": 1,
            "text": """
[TABLE]
DESCRIPTION | SKU | URL
--- | --- | ---
Firewall license | FG-7081F-BDL | https://example.invalid
[/TABLE]
""",
        }
    ]

    chunks = chunk_pages(pages, _doc_meta())
    spec_rows = [c for c in chunks if c["metadata"]["chunk_type"] == "spec_row"]

    assert spec_rows == []


def test_promotes_first_data_row_when_pdf_headers_are_generic():
    pages = [
        {
            "page": 1,
            "text": """
[TABLE]
Col0 | Col1 | Col2
--- | --- | ---
Specification | PA-3220 | PA-3250
Threat Prevention throughput | 2.4 Gbps | 3.8 Gbps
IPsec VPN throughput | 1.2 Gbps | 2 Gbps
[/TABLE]
""",
        }
    ]
    meta = _doc_meta(["PA-3220", "PA-3250"])
    meta["vendor"] = "Palo Alto"
    meta["family"] = "PA-3200 Series"
    meta["product_family"] = "PA-3200 Series"

    chunks = chunk_pages(pages, meta)
    spec_rows = [c for c in chunks if c["metadata"]["chunk_type"] == "spec_row"]

    assert len(spec_rows) == 4
    assert {c["metadata"]["model"] for c in spec_rows} == {"PA-3220", "PA-3250"}
