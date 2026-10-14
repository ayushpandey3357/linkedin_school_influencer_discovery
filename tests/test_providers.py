"""
Tests for data providers: Synthetic, CSV, and Authorized Connector.
"""

import io
import pytest
from app.config import AppConfig
from app.providers.authorized_connector import AuthorizedConnectorProvider
from app.providers.csv_provider import CSVProfileProvider
from app.providers.synthetic_provider import SyntheticDatasetProvider


def test_synthetic_provider_loads_benchmark_dataset():
    provider = SyntheticDatasetProvider()
    profiles = provider.get_profiles()

    # Must contain at least 25 synthetic education professionals
    assert len(profiles) >= 25, f"Expected at least 25 profiles, got {len(profiles)}"
    assert provider.get_total_count() >= 25

    # Check explicit labeling of all synthetic records
    for p in profiles:
        assert p.source_type == "SYNTHETIC TEST DATA"
        assert "example.com" in p.linkedin_url
        assert p.profile_id.startswith("SYN-")


def test_synthetic_provider_deduplication():
    provider = SyntheticDatasetProvider()
    profiles = provider.get_profiles()

    ids = [p.profile_id for p in profiles]
    urls = [p.linkedin_url for p in profiles]

    assert len(ids) == len(set(ids)), "Duplicate profile_id found in synthetic dataset"
    assert len(urls) == len(set(urls)), "Duplicate URL found in synthetic dataset"


def test_synthetic_provider_filtering():
    provider = SyntheticDatasetProvider()

    # Filter by role keyword
    principals = provider.get_profiles(filters={"role_keyword": "Principal"})
    assert len(principals) > 0
    for p in principals:
        assert "principal" in p.current_role.lower()

    # Filter by experience
    seniors = provider.get_profiles(filters={"min_experience": 20.0})
    assert len(seniors) > 0
    for p in seniors:
        assert p.experience_years >= 20.0


def test_csv_provider_loads_valid_csv():
    csv_data = """profile_id,name,linkedin_url,current_role,organization,location,experience_years
IMP-101,Aarav Patel,https://example.com/imported/aarav,Principal,St. Jude High School,Delhi,18
IMP-102,Bhavna Rao,https://example.com/imported/bhavna,Education Director,EduTrust India,Bengaluru,15
"""
    provider = CSVProfileProvider(source=io.StringIO(csv_data))
    profiles = provider.get_profiles()

    assert len(profiles) == 2
    assert profiles[0].name == "Aarav Patel"
    assert profiles[0].source_type == "IMPORTED DATA"
    assert profiles[1].organization == "EduTrust India"


def test_csv_provider_missing_required_column_raises_error():
    bad_csv = """profile_id,name,current_role
IMP-999,No URL User,Consultant
"""
    with pytest.raises(ValueError, match="missing mandatory columns"):
        CSVProfileProvider(source=io.StringIO(bad_csv))


def test_csv_provider_deduplication():
    duplicate_csv = """profile_id,name,linkedin_url,current_role
IMP-201,User One,https://example.com/imported/user-one,Director
IMP-201,User One Duplicate ID,https://example.com/imported/user-one-alt,Director
IMP-202,User Two,https://example.com/imported/user-two,Advisor
IMP-203,User Three Duplicate URL,https://example.com/imported/user-two,Advisor
"""
    provider = CSVProfileProvider(source=io.StringIO(duplicate_csv))
    profiles = provider.get_profiles()

    assert len(profiles) == 2
    assert profiles[0].profile_id == "IMP-201"
    assert profiles[1].profile_id == "IMP-202"
    assert len(provider.get_validation_errors()) >= 2


def test_csv_provider_loads_sample_csv_file():
    provider = CSVProfileProvider(AppConfig.SAMPLE_CSV_FILE)
    profiles = provider.get_profiles()
    assert len(profiles) >= 5
    assert all(p.source_type == "IMPORTED DATA" for p in profiles)


def test_authorized_connector_stub_behavior():
    connector = AuthorizedConnectorProvider()
    assert not connector.is_available()
    assert connector.get_total_count() == 0

    with pytest.raises(RuntimeError, match="not configured"):
        connector.get_profiles()
