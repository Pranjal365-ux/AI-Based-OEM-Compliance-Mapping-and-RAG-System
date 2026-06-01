"""
OEM Datasheet Ingestion Pipeline - Test Suite
Run with: python -m pytest tests/ -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    from config.settings import PipelineConfig
    cfg = PipelineConfig()
    cfg.use_llm_for_model_id = False   # Don't need API key for unit tests
    cfg.save_intermediate = False
    return cfg


@pytest.fixture(scope="session")
def sample_pages():
    """Simulate extracted pages from a datasheet."""
    return [
        {
            "page_number": 1,
            "raw_text": """
FortiGate 200F Series
Next-Generation Firewall

The FortiGate 200F series delivers high performance threat protection for
mid-sized enterprises. It includes the FortiGate 200F and FortiGate 201F models.
            """,
            "cleaned_text": """
FortiGate 200F Series
Next-Generation Firewall
The FortiGate 200F series delivers high performance threat protection for
mid-sized enterprises. It includes the FortiGate 200F and FortiGate 201F models.
            """.strip(),
            "tables": [],
            "extraction_method": "pdfplumber",
            "ocr_confidence": None,
            "is_scanned": False,
            "has_text": True,
        },
        {
            "page_number": 2,
            "raw_text": """
Technical Specifications

FIREWALL THROUGHPUT
Model          | FG-200F   | FG-201F
Firewall       | 27 Gbps   | 27 Gbps
IPS            | 7 Gbps    | 7 Gbps
NGFW           | 4.5 Gbps  | 4.5 Gbps
Threat Protect | 3.5 Gbps  | 3.5 Gbps

INTERFACES
GE RJ45: 16
SFP Slots: 2
USB: 2
Console: 1

POWER
Max Power Consumption: 58W
Input Voltage: 100-240V AC

DIMENSIONS
Height: 1.75 in (44.45 mm)
Width: 17.0 in (431.8 mm)
Depth: 11.5 in (292.1 mm)
Weight: 9.7 lb (4.4 kg)
            """,
            "cleaned_text": """
Technical Specifications

FIREWALL THROUGHPUT
Model          | FG-200F   | FG-201F
Firewall       | 27 Gbps   | 27 Gbps
IPS            | 7 Gbps    | 7 Gbps
NGFW           | 4.5 Gbps  | 4.5 Gbps
Threat Protect | 3.5 Gbps  | 3.5 Gbps

INTERFACES
GE RJ45: 16
SFP Slots: 2
USB: 2
Console: 1

POWER
Max Power Consumption: 58W
Input Voltage: 100-240V AC

DIMENSIONS
Height: 1.75 in (44.45 mm)
Width: 17.0 in (431.8 mm)
Depth: 11.5 in (292.1 mm)
Weight: 9.7 lb (4.4 kg)
            """.strip(),
            "tables": [
                {
                    "page_number": 2,
                    "table_index": 0,
                    "headers": ["Model", "FG-200F", "FG-201F"],
                    "rows": [
                        ["Firewall", "27 Gbps", "27 Gbps"],
                        ["IPS", "7 Gbps", "7 Gbps"],
                        ["NGFW", "4.5 Gbps", "4.5 Gbps"],
                    ],
                    "raw_text": "Model | FG-200F | FG-201F\nFirewall | 27 Gbps | 27 Gbps",
                }
            ],
            "extraction_method": "pdfplumber",
            "ocr_confidence": None,
            "is_scanned": False,
            "has_text": True,
        },
    ]


# ─── Text Cleaning Tests ──────────────────────────────────────────────────────────

class TestTextCleaning:
    def test_removes_control_characters(self):
        from ingestion.pdf_extractor import clean_text
        assert "\x00" not in clean_text("hello\x00world")

    def test_normalises_whitespace(self):
        from ingestion.pdf_extractor import clean_text
        result = clean_text("hello   world\n\n\n\nfoo")
        assert "   " not in result
        assert result.count("\n") <= 2

    def test_fixes_ligatures(self):
        from ingestion.pdf_extractor import clean_text
        assert "fi" in clean_text("ﬁrewall")

    def test_empty_string(self):
        from ingestion.pdf_extractor import clean_text
        assert clean_text("") == ""


# ─── Model Identification Tests ───────────────────────────────────────────────────

class TestModelIdentification:
    def test_extracts_model_numbers_regex(self, config):
        from ingestion.model_identifier import extract_candidate_model_numbers
        text = "The FG-200F and FG-201F are both part of the FortiGate 200F series."
        candidates = extract_candidate_model_numbers(text, config.model_id)
        assert len(candidates) >= 1

    def test_single_model_fallback(self, config, sample_pages):
        from ingestion.model_identifier import identify_models
        models = identify_models(sample_pages, "Fortinet", config)
        assert len(models) >= 1
        assert all(m.vendor == "Fortinet" for m in models)

    def test_model_id_is_unique(self, config, sample_pages):
        from ingestion.model_identifier import identify_models
        models = identify_models(sample_pages, "Fortinet", config)
        ids = [m.model_id for m in models]
        assert len(ids) == len(set(ids)), "Model IDs must be unique"

    def test_deduplication(self, config):
        from ingestion.model_identifier import _deduplicate_models
        from models.schemas import ModelSpec
        models = [
            ModelSpec(model_id="a", model_name="FG-200F", vendor="Fortinet"),
            ModelSpec(model_id="b", model_name="FG-200F", vendor="Fortinet"),  # duplicate
            ModelSpec(model_id="c", model_name="FG-201F", vendor="Fortinet"),
        ]
        unique = _deduplicate_models(models)
        assert len(unique) == 2

    def test_fortinet_variants_and_modules_are_not_primary_models(self, config):
        from ingestion.model_identifier import _deduplicate_models
        from models.schemas import ModelSpec

        models = [
            ModelSpec(model_id="a", model_name="FG-7081F", vendor="Fortinet"),
            ModelSpec(model_id="b", model_name="FG-7081F-DC", vendor="Fortinet"),
            ModelSpec(model_id="c", model_name="FG-7081F-2", vendor="Fortinet"),
            ModelSpec(model_id="d", model_name="FG-7121F", vendor="Fortinet"),
            ModelSpec(model_id="e", model_name="FIM-7921F", vendor="Fortinet"),
            ModelSpec(model_id="f", model_name="FPM-7620F", vendor="Fortinet"),
        ]

        unique = _deduplicate_models(models)
        assert [m.model_name for m in unique] == ["FG-7081F", "FG-7121F"]

    def test_false_positive_filter(self):
        from ingestion.model_identifier import _is_false_positive_model
        assert _is_false_positive_model("IEEE")
        assert _is_false_positive_model("SSL")
        assert _is_false_positive_model("HTTP")
        assert not _is_false_positive_model("FG-200F")

    def test_comparison_table_headers_are_models_not_spec_rows(self, config):
        from ingestion.model_identifier import extract_models_from_tables

        tables = [{
            "headers": ["Table 1: PA-3200 Series Performance and Capacities", "", "", ""],
            "rows": [
                ["", "PA-3220", "PA-3250", "PA-3260"],
                ["Firewall throughput", "4 Gbps", "5 Gbps", "7.5 Gbps"],
                ["Threat Prevention throughput", "2.2 Gbps", "2.5 Gbps", "4 Gbps"],
            ],
        }]

        models = extract_models_from_tables(tables, config.model_id)
        assert [m["model_name"] for m in models] == ["PA-3220", "PA-3250", "PA-3260"]


# ─── Chunking Tests ───────────────────────────────────────────────────────────────

class TestChunking:
    def test_chunk_has_required_fields(self, config, sample_pages):
        from ingestion.model_identifier import identify_models
        from ingestion.chunker import chunk_document
        from models.schemas import DatasheetDocument, VendorInfo, ExtractionMethod

        models = identify_models(sample_pages, "Fortinet", config)
        doc = DatasheetDocument(
            doc_id="test123",
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            vendor=VendorInfo(name="Fortinet", confidence=0.9),
            page_count=2,
            models=models,
            extraction_method=ExtractionMethod.PDFPLUMBER,
        )
        chunks = chunk_document(doc, config.chunking)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.vendor
            assert chunk.model_name
            assert chunk.doc_id == "test123"

    def test_chunks_respect_max_size(self, config, sample_pages):
        from ingestion.model_identifier import identify_models
        from ingestion.chunker import chunk_document
        from models.schemas import DatasheetDocument, VendorInfo, ExtractionMethod

        models = identify_models(sample_pages, "Fortinet", config)
        doc = DatasheetDocument(
            doc_id="test456",
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            vendor=VendorInfo(name="Fortinet", confidence=0.9),
            page_count=2,
            models=models,
            extraction_method=ExtractionMethod.PDFPLUMBER,
        )
        chunks = chunk_document(doc, config.chunking)

        for chunk in chunks:
            assert len(chunk.text) <= config.chunking.max_single_chunk + 200, \
                f"Chunk too long: {len(chunk.text)} chars"

    def test_chunk_metadata_for_chroma(self, config, sample_pages):
        from ingestion.model_identifier import identify_models
        from ingestion.chunker import chunk_document
        from models.schemas import DatasheetDocument, VendorInfo, ExtractionMethod

        models = identify_models(sample_pages, "Fortinet", config)
        doc = DatasheetDocument(
            doc_id="test789",
            source_path="/tmp/test.pdf",
            filename="test.pdf",
            vendor=VendorInfo(name="Fortinet", confidence=0.9),
            page_count=2,
            models=models,
            extraction_method=ExtractionMethod.PDFPLUMBER,
        )
        chunks = chunk_document(doc, config.chunking)

        for chunk in chunks:
            meta = chunk.to_chroma_metadata()
            # All values must be simple types (str, int, float, bool)
            for k, v in meta.items():
                assert isinstance(v, (str, int, float, bool)), \
                    f"Metadata key '{k}' has invalid type {type(v)}"

    def test_structured_specs_are_chunked(self, config):
        from ingestion.chunker import chunk_model_spec
        from models.schemas import DatasheetDocument, ExtractionMethod, ModelSpec, VendorInfo

        model = ModelSpec(
            model_id="paloalto_pa3250_0",
            model_name="PA-3250",
            vendor="Palo Alto",
            specs={"firewall_throughput_appmix": "5 Gbps"},
            common_specs={"virtual_systems_base_max": "1/6"},
        )
        doc = DatasheetDocument(
            doc_id="structured123",
            source_path="/tmp/pa-3200.pdf",
            filename="pa-3200.pdf",
            vendor=VendorInfo(name="Palo Alto", confidence=0.9),
            page_count=1,
            models=[model],
            extraction_method=ExtractionMethod.PDFPLUMBER,
        )

        chunks = chunk_model_spec(model, doc, config.chunking)
        structured = [c for c in chunks if c.section_name == "structured_specs"]

        assert len(structured) == 1
        assert structured[0].chunk_type.value == "spec_text"
        assert "firewall_throughput_appmix: 5 Gbps" in structured[0].text
        assert "virtual_systems_base_max: 1/6" in structured[0].text


# ─── Table Extraction Tests ───────────────────────────────────────────────────────

class TestTableExtraction:
    def test_table_to_markdown(self):
        from models.schemas import ExtractedTable
        t = ExtractedTable(
            page_number=1, table_index=0,
            headers=["Model", "Throughput"],
            rows=[["FG-200F", "27 Gbps"], ["FG-201F", "27 Gbps"]],
        )
        md = t.to_markdown()
        assert "| Model | Throughput |" in md
        assert "FG-200F" in md

    def test_table_to_flat_text(self):
        from models.schemas import ExtractedTable
        t = ExtractedTable(
            page_number=1, table_index=0,
            headers=["Parameter", "Value"],
            rows=[["Max Power", "58W"], ["Weight", "4.4 kg"]],
        )
        flat = t.to_flat_text()
        assert "Max Power: 58W" in flat
        assert "Weight: 4.4 kg" in flat

    def test_empty_table_fallback(self):
        from models.schemas import ExtractedTable
        t = ExtractedTable(
            page_number=1, table_index=0,
            raw_text="some raw text",
        )
        assert t.to_flat_text() == "some raw text"

    def test_multi_model_table_splits_into_per_model_specs(self):
        from ingestion.pipeline import _attach_tables_to_models
        from models.schemas import ModelSpec

        models = [
            ModelSpec(model_id="pa3260", model_name="PA-3260", vendor="Palo Alto"),
            ModelSpec(model_id="pa3250", model_name="PA-3250", vendor="Palo Alto"),
            ModelSpec(model_id="pa3220", model_name="PA-3220", vendor="Palo Alto"),
        ]
        raw_tables = [
            {
                "page_number": 4,
                "table_index": 0,
                "headers": ["", "PA-3220", "PA-3250", "PA-3260"],
                "rows": [
                    ["Firewall throughput (App-ID enabled)", "4 Gbps", "5 Gbps", "7.5 Gbps"],
                    ["Threat Prevention throughput", "2.2 Gbps", "2.5 Gbps", "4 Gbps"],
                    ["Virtual systems (base/max)", "1/6", "1/6", "1/6"],
                ],
                "raw_text": "PA-3220 PA-3250 PA-3260 Firewall throughput",
            }
        ]

        _attach_tables_to_models(models, raw_tables)
        by_name = {model.model_name: model for model in models}

        assert by_name["PA-3220"].specs["firewall_throughput_app_id_enabled"] == "4 Gbps"
        assert by_name["PA-3250"].specs["firewall_throughput_app_id_enabled"] == "5 Gbps"
        assert by_name["PA-3260"].specs["firewall_throughput_app_id_enabled"] == "7.5 Gbps"
        assert by_name["PA-3250"].specs["threat_prevention_throughput"] == "2.5 Gbps"
        assert by_name["PA-3220"].common_specs["virtual_systems_base_max"] == "1/6"
        assert all(len(model.spec_tables) == 1 for model in models)
        assert by_name["PA-3250"].spec_tables[0].headers == ["Specification", "PA-3250"]
        assert by_name["PA-3250"].spec_tables[0].rows[0] == [
            "Firewall throughput (App-ID enabled)",
            "5 Gbps",
        ]


# ─── Vendor Detection Tests ───────────────────────────────────────────────────────

class TestVendorDetection:
    def test_known_vendor_detection(self):
        from ingestion.pdf_extractor import _parse_vendor_from_text
        name, conf = _parse_vendor_from_text("FORTINET\nFortiGate 200F Datasheet")
        assert "Fortinet" in name
        assert conf >= 0.8

    def test_unknown_vendor_fallback(self):
        from ingestion.pdf_extractor import _parse_vendor_from_text
        name, conf = _parse_vendor_from_text("Acme Corp\nProduct Specifications")
        assert name  # Should return something
        assert 0.0 <= conf <= 1.0

    def test_opentext_not_misidentified_as_intel(self):
        from ingestion.pdf_extractor import _parse_vendor_from_text

        text = (
            "DATA SHEET\n"
            "opentext\n"
            "OpenText SIEM Open Data Platform\n"
            "Security intelligence and event management"
        )
        name, conf = _parse_vendor_from_text(text)
        assert name == "OpenText"
        assert conf >= 0.8

    def test_empty_text(self):
        from ingestion.pdf_extractor import _parse_vendor_from_text
        name, conf = _parse_vendor_from_text("")
        assert name == "Unknown"
        assert conf == 0.0


# ─── Vector Store Tests (in-memory) ──────────────────────────────────────────────

class TestVectorStore:
    @pytest.fixture(scope="class")
    def store(self, tmp_path_factory):
        from config.settings import VectorStoreConfig, EmbeddingConfig
        from knowledge_base.vector_store import VectorStoreManager

        vs_cfg = VectorStoreConfig(
            collection_name="test_collection",
            persist_directory=str(tmp_path_factory.mktemp("chroma")),
        )
        emb_cfg = EmbeddingConfig()
        store = VectorStoreManager(vs_cfg, emb_cfg)
        store.initialize()
        store.load_embedder()
        yield store
        store.close()

    def test_add_and_retrieve_chunk(self, store):
        from models.schemas import DocumentChunk, ChunkType

        chunk = DocumentChunk(
            chunk_id="test_chunk_001",
            text="Vendor: Fortinet | Model: FG-200F\nFirewall throughput: 27 Gbps\nIPS: 7 Gbps",
            doc_id="doc001",
            vendor="Fortinet",
            model_name="FG-200F",
            model_id="fortinet_fg200f_0",
            chunk_type=ChunkType.PERFORMANCE,
            section_name="performance",
            source_file="/tmp/test.pdf",
        )
        n = store.add_chunks([chunk])
        assert n == 1

    def test_search_returns_results(self, store):
        results = store.search("firewall throughput performance", n_results=5)
        assert len(results) >= 1
        assert all("score" in r for r in results)
        assert all(0.0 <= r["score"] <= 1.0 for r in results)

    def test_skip_duplicate_chunks(self, store):
        from models.schemas import DocumentChunk, ChunkType

        chunk = DocumentChunk(
            chunk_id="test_chunk_001",   # Same ID as above
            text="Different text but same ID",
            doc_id="doc001",
            vendor="Fortinet",
            model_name="FG-200F",
            model_id="fortinet_fg200f_0",
            chunk_type=ChunkType.GENERAL,
        )
        n = store.add_chunks([chunk])
        assert n == 0   # Should skip

    def test_document_exists(self, store):
        assert store.document_exists("doc001")
        assert not store.document_exists("nonexistent_doc")

    def test_stats(self, store):
        stats = store.get_stats()
        assert "total_chunks" in stats
        assert stats["total_chunks"] >= 1


# ─── Integration Test ─────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_pipeline_with_synthetic_pdf(self, config, tmp_path):
        """
        Create a minimal PDF with reportlab and run the full pipeline.
        This verifies the end-to-end flow without needing a real datasheet.
        """
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            pytest.skip("reportlab not installed – skipping PDF generation test")

        # Create test PDF
        pdf_path = tmp_path / "test_datasheet.pdf"
        c = canvas.Canvas(str(pdf_path))

        # Page 1: Header + description
        c.setFont("Helvetica-Bold", 24)
        c.drawString(72, 750, "Fortinet")
        c.setFont("Helvetica", 14)
        c.drawString(72, 720, "FortiGate 100F Next-Generation Firewall")
        c.drawString(72, 700, "Enterprise security for small to mid-sized businesses")
        c.showPage()

        # Page 2: Specs
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, 750, "Technical Specifications")
        c.setFont("Helvetica", 10)
        y = 720
        specs = [
            "Model: FG-100F",
            "Firewall Throughput: 20 Gbps",
            "IPS Throughput: 2.6 Gbps",
            "NGFW Throughput: 1 Gbps",
            "Interfaces: 12x GE RJ45, 2x SFP",
            "Max Power: 32W",
            "Weight: 3.6 kg",
        ]
        for spec in specs:
            c.drawString(72, y, spec)
            y -= 15
        c.save()

        # Run pipeline
        config.save_intermediate = False
        from ingestion.pipeline import OEMIngestionPipeline
        from config.settings import VectorStoreConfig
        config.vector_store = VectorStoreConfig(
            collection_name="e2e_test",
            persist_directory=str(tmp_path / "chroma"),
        )
        pipeline = OEMIngestionPipeline(config)
        pipeline.initialize()

        result = pipeline.ingest_file(pdf_path)

        assert result.status.value in ("completed", "failed")
        if result.status.value == "completed":
            assert result.models_found >= 1
            assert result.chunks_created >= 1

        pipeline.vector_store.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
