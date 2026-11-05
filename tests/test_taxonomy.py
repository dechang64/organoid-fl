# ── tests/test_taxonomy.py ──
"""Tests for the taxonomy registry — the single source of truth for labels.

These tests exist because the repository previously carried three mutually
contradictory label schemas and shipped a detector whose class list matched no
dataset. Most of what is asserted here is a regression guard against that
returning.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.taxonomy import (
    DEFAULT_TAXONOMY,
    LEGACY_TAXONOMY,
    DeprecatedTaxonomyWarning,
    LabelClass,
    Taxonomy,
    TaxonomyError,
    get_taxonomy,
    list_taxonomies,
    maybe_get_taxonomy,
    register_taxonomy,
    resolve_taxonomy,
    taxonomy_from_data_yaml,
    unregister_taxonomy,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBuiltinTaxonomies(unittest.TestCase):
    def test_default_is_the_schema_real_data_uses(self):
        """The default must match every dataset actually in the repository."""
        tax = get_taxonomy(DEFAULT_TAXONOMY)
        self.assertEqual(tax.name, "organoid_4class")
        self.assertEqual(tax.n_classes, 4)
        self.assertEqual(tax.keys, ("organoid0", "organoid1", "organoid3", "spheroid"))
        self.assertTrue(tax.is_dense)

    def test_legacy_three_class_schema_is_deprecated_not_deleted(self):
        """Old call sites must still resolve — but loudly."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tax = get_taxonomy(LEGACY_TAXONOMY)
        self.assertEqual(tax.keys, ("healthy", "early_stage", "late_stage"))
        self.assertTrue(tax.deprecated)
        self.assertEqual(tax.superseded_by, "developmental_4stage")
        self.assertTrue(
            any(issubclass(w.category, DeprecatedTaxonomyWarning) for w in caught),
            "requesting the legacy taxonomy must emit a DeprecationWarning",
        )

    def test_all_expected_taxonomies_are_registered(self):
        names = list_taxonomies()
        for expected in (
            "organoid_4class",
            "developmental_4stage",
            "multiorg_2class",
            "organoid_health_3class",
        ):
            self.assertIn(expected, names)

    def test_developmental_stages_carry_lineage(self):
        tax = get_taxonomy("developmental_4stage")
        self.assertEqual(
            tax.keys,
            ("cystic_non_budding", "early_budding", "late_budding", "spheroid"),
        )
        children = [c.key for c in tax.children_of("cystic_non_budding")]
        self.assertEqual(children, ["early_budding"])
        self.assertEqual(len(tax.roots()), 2)  # cystic root + spheroid root


class TestResolution(unittest.TestCase):
    def test_resolve_by_key_alias_label_and_id(self):
        tax = get_taxonomy("developmental_4stage")
        self.assertEqual(tax.resolve_key("early_budding"), "early_budding")
        self.assertEqual(tax.resolve_key("early"), "early_budding")       # alias
        self.assertEqual(tax.resolve_key("Early Budding"), "early_budding")  # label, any case
        self.assertEqual(tax.resolve_key(2), "late_budding")              # id
        self.assertEqual(tax.resolve_id("Early"), 1)

    def test_unknown_class_raises_with_a_useful_message(self):
        tax = get_taxonomy("multiorg_2class")
        with self.assertRaises(TaxonomyError) as ctx:
            tax.resolve_key("spheroid")
        self.assertIn("valid keys are", str(ctx.exception))

    def test_try_resolve_returns_none_instead_of_raising(self):
        tax = get_taxonomy("multiorg_2class")
        self.assertIsNone(tax.try_resolve_key("nope"))
        self.assertEqual(tax.try_resolve_key("macros"), "macros")

    def test_sparse_and_dense_index_validation(self):
        dense = get_taxonomy("organoid_4class")
        self.assertEqual(dense.validate_index(3), 3)
        with self.assertRaises(TaxonomyError):
            dense.validate_index(4)

        sparse = Taxonomy(
            name="sparse_demo",
            classes=(LabelClass(0, "a", "A"), LabelClass(2, "b", "B"), LabelClass(7, "c", "C")),
        )
        self.assertFalse(sparse.is_dense)
        self.assertEqual(sparse.id_order, (0, 2, 7))
        # positional mapping is opt-in, never implicit
        with self.assertRaises(TaxonomyError):
            sparse.validate_index(2)
        self.assertEqual(sparse.validate_index(2, allow_sparse=True), 7)


class TestTaxonomyConstruction(unittest.TestCase):
    def test_duplicate_ids_and_keys_are_rejected(self):
        with self.assertRaises(TaxonomyError):
            Taxonomy(name="dup_ids", classes=(LabelClass(0, "a", "A"), LabelClass(0, "b", "B")))
        with self.assertRaises(TaxonomyError):
            Taxonomy(name="dup_keys", classes=(LabelClass(0, "a", "A"), LabelClass(1, "a", "B")))

    def test_empty_taxonomy_is_rejected(self):
        with self.assertRaises(TaxonomyError):
            Taxonomy(name="empty", classes=())

    def test_unknown_parent_is_rejected(self):
        with self.assertRaises(TaxonomyError):
            Taxonomy(name="bad_parent", classes=(LabelClass(0, "a", "A", parent="ghost"),))

    def test_negative_id_is_rejected(self):
        with self.assertRaises(TaxonomyError):
            LabelClass(-1, "a", "A")

    def test_roundtrip_through_dict(self):
        original = get_taxonomy("developmental_4stage")
        restored = Taxonomy.from_dict(original.to_dict())
        self.assertEqual(restored.keys, original.keys)
        self.assertEqual(restored.n_classes, original.n_classes)

    def test_data_yaml_dict_export_is_yolo_shaped(self):
        payload = get_taxonomy("organoid_4class").to_data_yaml_dict()
        self.assertEqual(payload["nc"], 4)
        self.assertEqual(payload["names"][3], "spheroid")


class TestRegistryOperations(unittest.TestCase):
    def test_register_and_unregister(self):
        tax = Taxonomy(name="tmp_demo", classes=(LabelClass(0, "x", "X"),))
        register_taxonomy(tax)
        try:
            self.assertIs(get_taxonomy("tmp_demo"), tax)
            with self.assertRaises(TaxonomyError):
                register_taxonomy(tax)  # duplicate without overwrite
            register_taxonomy(tax, overwrite=True)
        finally:
            unregister_taxonomy("tmp_demo")
        self.assertIsNone(maybe_get_taxonomy("tmp_demo"))

    def test_unregister_unknown_raises(self):
        with self.assertRaises(TaxonomyError):
            unregister_taxonomy("never_registered")

    def test_get_unknown_lists_what_is_available(self):
        with self.assertRaises(TaxonomyError) as ctx:
            get_taxonomy("does_not_exist")
        self.assertIn("organoid_4class", str(ctx.exception))

    def test_name_lookup_is_case_and_separator_insensitive(self):
        self.assertEqual(get_taxonomy("Multiorg-2Class").name, "multiorg_2class")


class TestResolveTaxonomy(unittest.TestCase):
    def test_none_returns_default(self):
        self.assertEqual(resolve_taxonomy(None).name, DEFAULT_TAXONOMY)

    def test_accepts_instance_name_mapping_and_path(self):
        instance = get_taxonomy("multiorg_2class")
        self.assertIs(resolve_taxonomy(instance), instance)
        self.assertEqual(resolve_taxonomy("multiorg_2class").name, "multiorg_2class")

        mapped = resolve_taxonomy({"name": "inline", "names": ["p", "q"]})
        self.assertEqual(mapped.keys, ("p", "q"))

        yaml_path = os.path.join(REPO_ROOT, "data", "organoid_patches", "data.yaml")
        if os.path.isfile(yaml_path):
            imported = resolve_taxonomy(yaml_path)
            self.assertEqual(imported.n_classes, 4)

    def test_unresolvable_input_raises(self):
        with self.assertRaises(TaxonomyError):
            resolve_taxonomy("not_a_name_nor_a_file")


class TestDataYamlImport(unittest.TestCase):
    def test_real_datasets_import_to_four_classes(self):
        """The whole point: import what the data actually says."""
        candidates = [
            os.path.join(REPO_ROOT, "data", "organoid_patches", "data.yaml"),
            os.path.join(REPO_ROOT, "data", "intestinal_organoid", "OrganoidDataset", "data.yaml"),
            os.path.join(REPO_ROOT, "fl_detect_results", "client_data_iid", "client_0", "data.yaml"),
        ]
        checked = 0
        for path in candidates:
            if not os.path.isfile(path):
                continue
            tax = taxonomy_from_data_yaml(path, register=False)
            self.assertEqual(
                tax.keys, ("organoid0", "organoid1", "organoid3", "spheroid"), f"from {path}"
            )
            checked += 1
        self.assertGreater(checked, 0, "expected at least one real data.yaml in the repo")

    def test_multiorg_balanced_imports_to_two_classes(self):
        path = os.path.join(REPO_ROOT, "datasets", "multiorg_balanced", "data.yaml")
        if not os.path.isfile(path):
            self.skipTest("multiorg_balanced/data.yaml not present")
        tax = taxonomy_from_data_yaml(path, register=False)
        self.assertEqual(tax.n_classes, 2)
        self.assertIn("macros", [k.lower() for k in tax.keys])

    def test_missing_file_raises(self):
        with self.assertRaises(TaxonomyError):
            taxonomy_from_data_yaml(os.path.join(REPO_ROOT, "no_such_file.yaml"))

    def test_name_collision_with_different_classes_raises(self):
        first = taxonomy_from_data_yaml(
            os.path.join(REPO_ROOT, "data", "organoid_patches", "data.yaml")
        ) if os.path.isfile(os.path.join(REPO_ROOT, "data", "organoid_patches", "data.yaml")) else None
        if first is None:
            self.skipTest("organoid_patches/data.yaml not present")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(first.n_classes, 4)
        finally:
            # Leave the registry as we found it.
            unregister_taxonomy(first.name)


class TestConstantsShim(unittest.TestCase):
    def test_constants_no_longer_carry_the_placeholder_schema(self):
        from utils import constants

        self.assertEqual(constants.ORGANOID_CLASSES, ["organoid0", "organoid1", "organoid3", "spheroid"])
        self.assertEqual(constants.DEFAULT_NUM_CLASSES, 4)
        self.assertNotIn("healthy", constants.ORGANOID_CLASSES)

    def test_active_taxonomy_can_be_switched(self):
        from utils import constants

        original = constants.get_active_taxonomy()
        try:
            switched = constants.set_active_taxonomy("multiorg_2class")
            self.assertEqual(switched.n_classes, 2)
            self.assertIs(constants.get_active_taxonomy(), switched)
        finally:
            constants.set_active_taxonomy(original.name)

    def test_colors_cover_every_class(self):
        from utils import constants

        for key in constants.ORGANOID_CLASSES:
            self.assertIn(key, constants.COLORS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
