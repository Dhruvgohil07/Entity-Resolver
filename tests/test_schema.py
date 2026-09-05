"""Tests for the canonical Record model."""

import pytest
from pydantic import ValidationError

from dedup.schema import Record


def _full_record(**overrides: object) -> Record:
    fields = {
        "record_id": "abt_buy:10",
        "source": "abt_buy",
        "entity_id": "entity-1",
        "title": "Sony Widget 3000",
        "description": "A widget with knobs.",
        "brand": "Sony",
        "category": "Widgets",
        "price": 19.99,
        "raw_attributes": {"upc": "012345"},
    }
    fields.update(overrides)
    return Record(**fields)


def test_round_trip_serialization():
    record = _full_record()
    assert Record.model_validate(record.model_dump()) == record


def test_minimal_construction_defaults():
    record = Record(record_id="abt_buy:1", source="abt_buy", title="Widget")
    assert record.entity_id is None
    assert record.description is None
    assert record.brand is None
    assert record.category is None
    assert record.price is None
    assert record.raw_attributes == {}

    other = Record(record_id="abt_buy:2", source="abt_buy", title="Widget 2")
    assert record.raw_attributes is not other.raw_attributes


def test_source_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Record(record_id="dblp:1", source="dblp", title="Paper")


@pytest.mark.parametrize("field", ["record_id", "title"])
def test_empty_required_string_rejected(field):
    with pytest.raises(ValidationError):
        _full_record(**{field: ""})


def test_price_negative_rejected():
    with pytest.raises(ValidationError):
        _full_record(price=-1.0)


def test_price_zero_and_none_accepted():
    assert _full_record(price=0.0).price == 0.0
    assert _full_record(price=None).price is None


def test_compound_record_id_convention_keeps_sources_distinct():
    abt = _full_record(record_id="abt_buy:10", source="abt_buy")
    google = _full_record(record_id="amazon_google:10", source="amazon_google")
    assert abt != google
    assert abt.record_id != google.record_id


def test_description_none_vs_empty_string_distinguishable():
    absent = _full_record(description=None)
    present_empty = _full_record(description="")
    assert absent.description is None
    assert present_empty.description == ""
    assert absent != present_empty


def test_entity_id_serve_time_vs_benchmark_shape():
    serve_time = _full_record(entity_id=None)
    benchmark = _full_record(entity_id="entity-1")
    assert serve_time.entity_id is None
    assert benchmark.entity_id == "entity-1"
