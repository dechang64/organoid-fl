# ── tests/test_pipeline_review.py ──
"""M2 acceptance: the read path, disagreement analysis, supplements, strata.

The three exit criteria of PRD 10.4 M2 are behavioural, not structural, so each
one is asserted on the payload a viewer actually receives or on the record the
workflow actually writes:

* P-01 — no model pre-annotation and no peer revision in any blind payload, for
  every role, checked as *absent keys* rather than empty sections.
* P-02 — the analysis exists by the time the third submission returns, and its
  timestamp is the one the workflow was given.
* P-06 — the blind revision's content hash survives a supplement.

The metric layer gets its own tests because "α agrees with a standard tool within
0.001" (PRD 4.5) cannot be checked against the implementation itself: one case is
computed by hand in the test.
"""

from __future__ import annotations

import json

import pytest

from analysis.contracts.enums import Role
from analysis.contracts.protocol import make_protocol
from analysis.pipeline import AnnotationService, SQLiteStore
from analysis.pipeline.disagreement import (
    DisagreementError,
    compute_disagreement,
    decode_region_map,
)
from analysis.pipeline.geometry import (
    CoderPayload,
    GeometryError,
    bootstrap_alpha_ci,
    build_units,
    hausdorff95,
    krippendorff_alpha_nominal,
    match_payloads,
    positive_negative_agreement,
    rasterize_polygon,
    raw_agreement,
)
from analysis.pipeline.stratum import StratumRuleError, label_stratum

READERS = [("expert_a", Role.DOMAIN_EXPERT), ("expert_b", Role.DOMAIN_EXPERT), ("expert_c", Role.DOMAIN_EXPERT)]
SIGNALS = {"contrast": 0.62, "object_count": 6, "clustered_fraction": 0.05, "size_cv": 0.2}
T0 = "2026-11-05T09:00:00Z"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def payload(instances, *, width=200, height=200) -> bytes:
    return json.dumps(
        {"format": "organoid-geometry-v1", "width": width, "height": height, "instances": instances}
    ).encode("utf-8")


def square(instance_id="i1", *, x=10, y=10, size=40, class_key="organoid", confidence=None):
    instance = {"id": instance_id, "class_key": class_key, "rect": [x, y, x + size, y + size]}
    if confidence is not None:
        instance["confidence"] = confidence
    return instance


def protocol():
    return make_protocol(
        id="proto-1",
        version="1.0",
        taxonomy_name="organoid_4class",
        touching_object_rule="separate_if_boundary_visible",
        min_visible_boundary_um=2.0,
        overlap_split_rule="split_at_waist",
        effective_from="2026-01-01T00:00:00Z",
    )


def make_service(path=":memory:") -> AnnotationService:
    return AnnotationService(SQLiteStore(path))


def ready_task(svc: AnnotationService, *, samples=("S-001",), template="research") -> None:
    """Create, label and publish a task. Research template requires strata."""
    svc.create_task(
        task_id="T1",
        node_id="node_a",
        project_id="proj",
        template=template,
        image_scope=samples,
        protocol_id="proto-1",
        protocol=protocol(),
        now=T0,
    )
    for sample_id in samples:
        svc.label_stratum(
            "T1", sample_id, signals=SIGNALS, allow_provisional=True, now=T0
        )
    svc.publish("T1", actor_id="owner_1", readers=READERS, now=T0)


def submit_three(svc: AnnotationService, *, sample_id="S-001", c_payload=None) -> None:
    svc.submit_blind("T1", sample_id, actor_id="expert_a", payload=payload([square("a1")]), now=T0)
    svc.submit_blind(
        "T1", sample_id, actor_id="expert_b", payload=payload([square("b1", x=12)]), now=T0
    )
    svc.submit_blind(
        "T1",
        sample_id,
        actor_id="expert_c",
        payload=c_payload or payload([square("c1", class_key="cyst")]),
        now=T0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Payload and geometry
# ─────────────────────────────────────────────────────────────────────────────


def test_payload_reader_refuses_what_it_cannot_measure():
    with pytest.raises(GeometryError, match="format"):
        CoderPayload.parse(payload([], width=200).replace(b"organoid-geometry-v1", b"v0"))
    with pytest.raises(GeometryError, match="unknown top-level"):
        CoderPayload.parse(
            json.dumps(
                {"format": "organoid-geometry-v1", "width": 10, "height": 10, "instances": [], "scale": 2}
            ).encode()
        )
    # An out-of-canvas point is refused rather than clamped: clamping moves the
    # annotation and every downstream metric measures the moved version.
    with pytest.raises(GeometryError, match="outside"):
        CoderPayload.parse(payload([square(x=190, size=40)]))
    with pytest.raises(GeometryError, match="at least 3 points"):
        CoderPayload.parse(payload([{"id": "i", "class_key": "organoid", "polygon": [[0, 0], [1, 1]]}]))
    # A zero-area object is a data error, not an empty annotation.
    with pytest.raises(GeometryError, match="empty mask"):
        CoderPayload.parse(
            payload([{"id": "i", "class_key": "organoid", "polygon": [[10, 10], [10, 10], [10, 10]]}])
        )


def test_rasterisation_covers_the_rectangle_it_describes():
    parsed = CoderPayload.parse(payload([square(size=40)]))
    assert int(parsed.mask("i1").sum()) == 1600  # 40 × 40, pixel centres
    triangle = rasterize_polygon([(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)], 20, 20)
    assert 150 <= int(triangle.sum()) <= 210  # 200 px ideal, half-pixel edge


def test_matching_is_one_to_one_and_thresholded():
    a = CoderPayload.parse(payload([square("a1"), square("a2", x=100)]))
    b = CoderPayload.parse(
        payload([square("b1", x=11), square("b2", x=100), square("b3", x=160)])
    )
    result = match_payloads(a, b)
    assert {p.a for p in result.pairs} == {"a1", "a2"}
    assert len(result.pairs) == 2  # b3 has no counterpart → FP, never a duplicate match
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 0
    assert result.detection_f1 == pytest.approx(2 * 2 / (2 * 2 + 1))

    far = CoderPayload.parse(payload([square("c1", x=190, y=190, size=9)]))
    assert match_payloads(a, far).pairs == ()


def test_hausdorff_is_zero_on_identical_masks_and_counts_the_shift():
    a = CoderPayload.parse(payload([square()]))
    same = CoderPayload.parse(payload([square()]))
    assert hausdorff95(a.mask("i1"), same.mask("i1")) == (0.0, 0.0)

    shifted = CoderPayload.parse(payload([square(x=13)]))
    hd95, assd = hausdorff95(a.mask("i1"), shifted.mask("i1"))
    assert hd95 == pytest.approx(3.0)  # a 3-pixel shift, measured in pixels
    # Half the pooled surface distances are 3 and half are 0 (the leading and
    # trailing edges of the two squares coincide in y), so the mean is 1.5.
    assert assd == pytest.approx(1.5)


def test_dice_matches_the_closed_form_for_a_known_overlap():
    a = CoderPayload.parse(payload([square(size=40)]))
    b = CoderPayload.parse(payload([square(x=20, size=40)]))
    result = match_payloads(a, b)
    # Two 40×40 squares displaced by 20 px: intersection 30×40 = 1200, union 2000.
    assert result.pairs[0].dice == pytest.approx(2 * 1200 / 3200)
    assert result.pairs[0].iou == pytest.approx(1200 / 2000)


# ─────────────────────────────────────────────────────────────────────────────
# Attribute agreement
# ─────────────────────────────────────────────────────────────────────────────


def test_krippendorff_alpha_matches_a_hand_computed_case():
    """Four units, three coders, one disagreement — computed by hand.

    Coincidence: o_xx = 4, o_yy = 6, o_xy = o_yx = 1 → n = 12, Do = 2/12.
    Marginals: x = 5, y = 7 → De = (5·7 + 7·5)/(12·11) = 70/132.
    α = 1 − (2/12)/(70/132) = 0.685714…
    """
    units = [["x", "x", "x"], ["x", "x", "y"], ["y", "y", "y"], ["y", "y", "y"]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(1 - (2 / 12) / (70 / 132), abs=1e-12)


def test_alpha_refuses_to_invent_a_number_from_too_little_data():
    assert krippendorff_alpha_nominal([["x", "x"]]) is None  # one pairable unit
    # A unit with one value is dropped, but two agreeing units still carry α = 1.
    assert krippendorff_alpha_nominal([["x", "x"], ["y", None], ["y", "y"]]) == pytest.approx(1.0)
    # Systematic disagreement: Do = 1, De = 2/3, so α = −0.5.
    assert krippendorff_alpha_nominal([["x", "y"], ["y", "x"]]) == pytest.approx(-0.5)


def test_bootstrap_interval_is_reproducible_for_a_fixed_seed():
    units = [["a", "a"], ["a", "b"], ["b", "b"], ["b", "b"], ["a", "b"]]
    first = bootstrap_alpha_ci(units, resamples=200, seed=7)
    second = bootstrap_alpha_ci(units, resamples=200, seed=7)
    assert first == second
    assert first["ci_low"] <= first["alpha"] <= first["ci_high"]
    assert first["seed"] == 7 and first["resamples"] == 200


def test_raw_and_positive_negative_agreement_disagree_when_one_class_dominates():
    """The Kappa paradox PRD 6.1 asks to be reported separately."""
    units = [["a", "a"], ["a", "a"], ["a", "b"], ["a", "a"], ["a", "a"]]
    assert raw_agreement(units) == pytest.approx(4 / 5)
    scores = positive_negative_agreement(units)
    # "a": 4 co-positive pairs, 1 discordant → 2·4/(2·4+1) = 0.889.
    # "b": never co-positive → 0. Macro over the two classes = 0.444, which is
    # why the per-class values are reported next to it.
    assert scores["per_class"]["a"]["positive"] == pytest.approx(2 * 4 / (2 * 4 + 0 + 1))
    assert scores["per_class"]["b"]["positive"] == pytest.approx(0.0)
    assert scores["macro_positive"] == pytest.approx((8 / 9 + 0) / 2)
    # No unit is negative for "a" (every unit has an "a" in it), yet the one
    # discordant pair still puts "a" in the denominator — so this is 0.0, not
    # "undefined". Reporting it as 1.0 would claim agreement on an empty set.
    assert scores["per_class"]["a"]["negative"] == pytest.approx(0.0)
    assert scores["per_class"]["a"]["co_negative_pairs"] == 0


def test_units_keep_a_missing_coder_missing_instead_of_guessing():
    payloads = {
        "a": CoderPayload.parse(payload([square("a1"), square("a2", x=100)])),
        "b": CoderPayload.parse(payload([square("b1", x=11)])),
    }
    units = build_units(payloads)
    assert len(units) == 2
    values = [dict(unit.values) for unit in units]
    assert {"a": "organoid", "b": "organoid"} in values
    assert {"a": "organoid"} in values  # b did not annotate the second object


# ─────────────────────────────────────────────────────────────────────────────
# Disagreement analysis
# ─────────────────────────────────────────────────────────────────────────────


def test_disagreement_needs_two_coders():
    with pytest.raises(DisagreementError, match="at least two"):
        compute_disagreement("S-001", {"a": payload([square()])})


def test_identical_products_score_zero_and_different_ones_score_higher():
    # Two objects, because one object yields a single unit and α is then
    # undefined by design — a zero controversy score here has to come from
    # measured agreement, not from having nothing to measure.
    objects = [square("i1"), square("i2", x=100)]
    same = {coder: payload(objects) for coder in ("a", "b", "c")}
    identical = compute_disagreement("S-001", same)
    assert identical.controversy_score == pytest.approx(0.0)
    assert identical.snapshot["alpha"]["alpha"] == pytest.approx(1.0)

    split = {
        "a": payload(objects),
        "b": payload([square("i1", x=12), square("i2", x=100)]),
        "c": payload([square("i1", class_key="cyst"), square("i2", x=100)]),
    }
    disputed = compute_disagreement("S-001", split)
    assert disputed.controversy_score > identical.controversy_score


def test_absent_components_are_named_and_the_weights_renormalised():
    """No pre-annotation and no confidence: the score must not silently assume zero."""
    data = {
        "a": payload([square("i1"), square("i2", x=100)]),
        "b": payload([square("i1", x=12), square("i2", x=100)]),
    }
    result = compute_disagreement("S-001", data)
    controversy = result.snapshot["controversy"]
    assert controversy["components"]["preannotation"] is None
    assert "preannotation" in controversy["missing_components"]
    assert set(controversy["weights"]) == {"iou", "attribute"}
    assert sum(controversy["weights"].values()) == pytest.approx(1.0)
    assert result.controversy_score == pytest.approx(
        sum(w * controversy["components"][k] for k, w in controversy["weights"].items())
    )


def test_confidence_is_used_when_supplied():
    with_conf = {
        "a": payload([square("i1", confidence=1.0), square("i2", x=100, confidence=1.0)]),
        "b": payload([square("i1", confidence=1.0), square("i2", x=100, confidence=1.0)]),
    }
    low_conf = {
        "a": payload([square("i1", confidence=0.1), square("i2", x=100, confidence=0.1)]),
        "b": payload([square("i1", confidence=0.1), square("i2", x=100, confidence=0.1)]),
    }
    assert compute_disagreement("S", with_conf).snapshot["controversy"]["components"]["confidence"] == 0.0
    assert compute_disagreement("S", low_conf).snapshot["controversy"]["components"][
        "confidence"
    ] == pytest.approx(0.9)


def test_region_map_round_trips_to_the_votes_it_claims():
    data = {
        "a": payload([square()]),
        "b": payload([square(x=20)]),
    }
    result = compute_disagreement("S", data)
    votes = decode_region_map(result.regions)
    assert votes.shape == (200, 200)
    # 20 px overlap: that band is covered twice, the rest once.
    assert int(votes.max()) == 2
    assert int((votes == 2).sum()) == 30 * 40  # 20..50 is 30 px wide
    assert int((votes > 0).sum()) == 50 * 40  # 10..60
    # With two coders a pixel is agreed only where both voted, so the disputed
    # share is the non-overlapping part: 800 of 2000 covered pixels.
    assert result.snapshot["regions"]["disputed_share"] == pytest.approx(800 / 2000)


def test_revision_set_hash_changes_when_a_payload_changes():
    base = {"a": payload([square()]), "b": payload([square()])}
    changed = {"a": payload([square(x=1)]), "b": payload([square()])}
    assert compute_disagreement("S", base).revision_set_hash != compute_disagreement(
        "S", changed
    ).revision_set_hash


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion: strata (D-08, Q-07)
# ─────────────────────────────────────────────────────────────────────────────


def test_research_publish_refuses_a_sample_without_a_stratum():
    svc = make_service()
    svc.create_task(
        task_id="T1",
        node_id="node_a",
        project_id="proj",
        template="research",
        image_scope=("S-001",),
        protocol_id="proto-1",
        protocol=protocol(),
    )
    with pytest.raises(Exception, match="requires a difficulty stratum"):
        svc.publish("T1", actor_id="owner_1", readers=READERS)


def test_provisional_rule_requires_an_explicit_acknowledgement():
    with pytest.raises(StratumRuleError, match="Q-07"):
        label_stratum("S-001", SIGNALS)
    label = label_stratum("S-001", SIGNALS, allow_provisional=True)
    assert label.rule_version == "provisional_v1"
    assert label.signal_snapshot["rule_provisional"] is True


def test_a_missing_signal_fails_ingestion_rather_than_defaulting_to_easy():
    incomplete = {"contrast": 0.6, "object_count": 3}
    with pytest.raises(StratumRuleError, match="missing signal"):
        label_stratum("S-001", incomplete, allow_provisional=True)


def test_stratum_cannot_be_assigned_after_the_sample_is_being_read():
    svc = make_service()
    ready_task(svc)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=payload([square("a1")]), now=T0)
    with pytest.raises(Exception, match="assigned at ingestion"):
        svc.label_stratum("T1", "S-001", signals=SIGNALS, allow_provisional=True)


def test_teaching_template_does_not_require_a_stratum():
    svc = make_service()
    svc.create_task(
        task_id="T1",
        node_id="node_a",
        project_id="proj",
        template="teaching",
        image_scope=("S-001",),
        protocol_id="proto-1",
        protocol=protocol(),
    )
    svc.publish(
        "T1", actor_id="owner_1", readers=[("student_1", Role.ANNOTATOR)], now=T0
    )
    assert svc.stratum("S-001") is None


# ─────────────────────────────────────────────────────────────────────────────
# P-02: the analysis is triggered by the workflow
# ─────────────────────────────────────────────────────────────────────────────


def test_third_submission_triggers_the_analysis_with_no_extra_call():
    svc = make_service()
    ready_task(svc)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=payload([square("a1")]), now=T0)
    svc.submit_blind("T1", "S-001", actor_id="expert_b", payload=payload([square("b1", x=12)]), now=T0)
    assert svc.read_back("T1")["disagreements"] == []

    svc.submit_blind(
        "T1", "S-001", actor_id="expert_c", payload=payload([square("c1")]), now=T0
    )
    record = svc.read_back("T1")["disagreements"][0]
    assert record.sample_id == "S-001"
    assert record.computed_at == T0  # the timestamp the workflow was given
    assert record.metric_snapshot["triggered_by"] == "auto"
    assert 0.0 <= record.controversy_score <= 1.0
    assert record.metric_snapshot["coders"] == ["expert_a", "expert_b", "expert_c"]
    assert record.metric_snapshot["metrics_version"] == "organoid-metrics-v1"


def test_the_analysis_is_not_recomputed_when_nothing_changed():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    first = svc.read_back("T1")["disagreements"]
    again = svc.compute_disagreement("T1", "S-001", triggered_by="manual")
    third = svc.read_back("T1")["disagreements"]
    assert again.id == first[0].id
    assert len(third) == 1
    assert [op for op in svc.chain.operations()].count("disagreement_computed") == 1


def test_a_supplement_does_not_change_the_measured_set():
    """The metric set stays the blind reads; a supplement is a dependent observation."""
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    svc.reveal("T1", "S-001", now=T0)
    before = svc.read_back("T1")["disagreements"][0]
    svc.supplement("T1", "S-001", actor_id="expert_a", payload=payload([square("a2", x=5)]), now=T0)
    after = svc.read_back("T1")["disagreements"]
    assert len(after) == 1
    assert after[0].id == before.id
    assert after[0].metric_snapshot["coders"] == ["expert_a", "expert_b", "expert_c"]


# ─────────────────────────────────────────────────────────────────────────────
# P-01: the projected read path
# ─────────────────────────────────────────────────────────────────────────────


FORBIDDEN_WHILE_BLIND = {"model_rank", "disagreement", "adjudication", "cosign", "preannotation"}
FORBIDDEN_FOR_EXPERT_ALWAYS = {"model_rank", "adjudication", "cosign"}


def test_blind_payload_has_no_forbidden_section_for_any_role():
    svc = make_service()
    ready_task(svc)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=payload([square("a1")]), now=T0)
    svc.submit_blind("T1", "S-001", actor_id="expert_b", payload=payload([square("b1")]), now=T0)

    for role, who in (
        (Role.DOMAIN_EXPERT, "expert_a"),
        (Role.CHIEF_EXPERT, "chief_1"),
        (Role.PROJECT_OWNER, "owner_1"),
        (Role.NODE_ADMIN, "admin_1"),
        (Role.COSIGNER, "cosigner_1"),
    ):
        view = svc.view_task("T1", viewer_role=role, viewer_id=who)
        sample = view["samples"][0]
        assert sample["stage"] == "blind"
        assert not (FORBIDDEN_WHILE_BLIND & set(sample)), f"{role.value} leaked {FORBIDDEN_WHILE_BLIND & set(sample)}"
        # The keys are absent, not None: a client that renders a None section
        # learns to tolerate it, and one forgotten branch later it renders.
        assert all(sample.get(key) is None for key in FORBIDDEN_WHILE_BLIND)


def test_an_expert_sees_only_their_own_revision_while_blind():
    svc = make_service()
    ready_task(svc)
    svc.submit_blind("T1", "S-001", actor_id="expert_a", payload=payload([square("a1")]), now=T0)
    svc.submit_blind("T1", "S-001", actor_id="expert_b", payload=payload([square("b1")]), now=T0)

    view = svc.view_task("T1", viewer_role=Role.DOMAIN_EXPERT, viewer_id="expert_a")
    authors = [r["author_id"] for r in view["samples"][0]["revisions"]]
    assert authors == ["expert_a"]


def test_chief_sees_the_analysis_and_the_expert_still_does_not():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    svc.reveal("T1", "S-001", now=T0)

    chief = svc.view_task("T1", viewer_role=Role.CHIEF_EXPERT, viewer_id="chief_1")["samples"][0]
    assert chief["stage"] == "revealed"
    assert "disagreement" in chief
    assert len(chief["revisions"]) == 3  # PRD 3.2: 首席 sees 三份 after the analysis

    expert = svc.view_task("T1", viewer_role=Role.DOMAIN_EXPERT, viewer_id="expert_a")["samples"][0]
    # PRD 3.2's 「否」 wins over PRD 4.5's "three revisions to three coders"
    # until that conflict is resolved (see analysis/pipeline/README.md).
    assert [r["author_id"] for r in expert["revisions"]] == ["expert_a"]
    assert not (FORBIDDEN_FOR_EXPERT_ALWAYS & set(expert))
    assert "disagreement" not in expert


def test_view_refuses_a_task_without_a_protocol():
    svc = make_service()
    svc.create_task(
        task_id="T1",
        node_id="node_a",
        project_id="proj",
        template="teaching",
        image_scope=("S-001",),
        protocol_id="proto-1",
    )
    with pytest.raises(Exception, match="no stored protocol"):
        svc.view_task("T1", viewer_role=Role.PROJECT_OWNER, viewer_id="owner_1")


# ─────────────────────────────────────────────────────────────────────────────
# P-06: supplements
# ─────────────────────────────────────────────────────────────────────────────


def test_supplement_leaves_the_blind_revision_hash_unchanged():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    blind_before = {
        r.id: r.payload_hash for r in svc.read_back("T1")["revisions"] if r.author_id == "expert_a"
    }
    svc.reveal("T1", "S-001", now=T0)
    supplement = svc.supplement(
        "T1", "S-001", actor_id="expert_a", payload=payload([square("a2", x=5)]), now=T0
    )
    after = svc.read_back("T1")["revisions"]
    blind_after = {r.id: r.payload_hash for r in after if r.author_id == "expert_a" and r.is_blind_read}
    assert blind_after == blind_before
    assert supplement.derived_from in blind_before
    assert supplement.is_blind_read is False
    assert supplement.revision_no == 2


def test_only_one_supplement_and_only_after_the_reveal():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    with pytest.raises(Exception, match=r"after the reveal|awaiting_reveal"):
        svc.supplement("T1", "S-001", actor_id="expert_a", payload=payload([square("a2")]), now=T0)
    svc.reveal("T1", "S-001", now=T0)
    svc.supplement("T1", "S-001", actor_id="expert_a", payload=payload([square("a2")]), now=T0)
    with pytest.raises(Exception, match="already has a supplement"):
        svc.supplement("T1", "S-001", actor_id="expert_a", payload=payload([square("a3")]), now=T0)


def test_a_coder_who_was_not_assigned_cannot_supplement():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    svc.reveal("T1", "S-001", now=T0)
    with pytest.raises(Exception, match="holds no assignment"):
        svc.supplement("T1", "S-001", actor_id="expert_z", payload=payload([square()]), now=T0)


# ─────────────────────────────────────────────────────────────────────────────
# Metric matrix
# ─────────────────────────────────────────────────────────────────────────────


def test_metric_matrix_has_a_row_for_every_sample_including_unfinished_ones():
    svc = make_service()
    ready_task(svc, samples=("S-001", "S-002"))
    submit_three(svc, sample_id="S-001")
    matrix = svc.metric_matrix("T1")
    assert [row["sample_id"] for row in matrix["rows"]] == ["S-001", "S-002"]
    assert matrix["rows"][0]["disagreement_id"] is not None
    assert matrix["rows"][1]["disagreement_id"] is None
    # Unfinished samples report None rather than 0: an average that counts them
    # as zero reads as a bad score and sends somebody looking for a modelling bug.
    assert matrix["rows"][1]["dice_mean"] is None
    assert matrix["summary"]["with_disagreement"] == 1


def test_matrix_reports_the_metrics_the_prd_names():
    svc = make_service()
    ready_task(svc)
    submit_three(svc)
    row = svc.metric_matrix("T1")["rows"][0]
    for key in ("alpha", "alpha_ci", "raw_agreement", "dice_mean", "detection_f1", "hd95_mean", "controversy_score"):
        assert key in row
    assert row["stratum"] == "easy"
    assert row["stratum_rule"] == "provisional_v1"
    assert row["dice_mean"] == pytest.approx(1.0, abs=0.05)  # same square, 2 px apart
