"""Unit tests for split auditing, source-video isolation, and cross-protocol leakage detection."""

import pytest

from sentinel_vl.data.schemas import SplitPartition, ValidationError
from sentinel_vl.data.split_auditor import SplitAuditor


def test_split_partition_clean(clean_partition: SplitPartition) -> None:
    clean_partition.validate_disjointness()  # Should not raise
    audit_res = SplitAuditor.audit_partition(clean_partition)
    assert audit_res.is_valid is True
    assert len(audit_res.internal_leaks) == 0
    assert len(audit_res.errors) == 0


def test_split_partition_internal_leak(leaking_partition: SplitPartition) -> None:
    with pytest.raises(ValidationError, match="Data leakage detected in partition"):
        leaking_partition.validate_disjointness()

    audit_res = SplitAuditor.audit_partition(leaking_partition)
    assert audit_res.is_valid is False
    assert "train_vs_val" in audit_res.internal_leaks
    assert audit_res.internal_leaks["train_vs_val"] == ["vid_03"]


def test_cross_protocol_leak_detection() -> None:
    protocol_retrieval = SplitPartition(
        name="uca_retrieval",
        train_video_ids={"vid_A", "vid_B", "vid_C"},
        val_video_ids={"vid_D"},
        test_video_ids={"vid_E"},
    )

    # Clean cross-protocol: anomaly evaluation set has no overlap with retrieval train
    protocol_anomaly_clean = SplitPartition(
        name="ucf_anomaly",
        train_video_ids={"vid_A", "vid_B"},
        val_video_ids={"vid_F"},
        test_video_ids={"vid_G"},
    )

    res_clean = SplitAuditor.audit_cross_protocol(protocol_retrieval, protocol_anomaly_clean)
    assert res_clean.is_valid is True
    assert len(res_clean.cross_protocol_leaks) == 0

    # Contaminated cross-protocol: vid_C is trained in retrieval, but evaluated in anomaly test!
    protocol_anomaly_contaminated = SplitPartition(
        name="ucf_anomaly",
        train_video_ids={"vid_A"},
        val_video_ids={"vid_F"},
        test_video_ids={"vid_C"},  # Contamination!
    )

    res_leak = SplitAuditor.audit_cross_protocol(protocol_retrieval, protocol_anomaly_contaminated)
    assert res_leak.is_valid is False
    assert "uca_retrieval_train_in_ucf_anomaly_test" in res_leak.cross_protocol_leaks
    assert res_leak.cross_protocol_leaks["uca_retrieval_train_in_ucf_anomaly_test"] == ["vid_C"]

