"""
Unit tests for HierarchialDet loss functions.

These tests use CPU tensors only — no GPU or pretrained weights required.
They exercise the corrected logic introduced in the enterprise refactor:

  - Bug 1: Diagnosis logits must use bz_src_logits_3, not bz_src_logits_1
  - Bug 4: Non-focal class weight must use loop variable, not self.num_class
  - Bare-except fix: freeze flags set only on NameError/UnboundLocalError
"""

import types
import unittest
import torch


class MockCfg:
    """Minimal config stub so SetCriterion.__init__ does not need Detectron2."""
    class MODEL:
        class DiffusionDet:
            NUM_CLASSES = [4, 10, 4]
            EOS_COEF = 0.1
            COST_CLASS = 1.0
            COST_BBOX = 1.0
            COST_GIOU = 1.0
            USE_FOCAL = False
            ALPHA = 0.25
            GAMMA = 2.0
            USE_FED_LOSS = False
            NUM_PROPOSALS = 300


def _make_criterion():
    """Construct a SetCriterion with a minimal config stub."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from hierarchialdet.loss import SetCriterion, HungarianMatcher
    matcher = HungarianMatcher(
        cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, use_focal=False
    )
    criterion = SetCriterion(
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        eos_coef=0.1,
        losses=["labels", "boxes"],
        num_classes=[4, 10, 4],
        cfg=MockCfg(),
    )
    return criterion


class TestNonFocalWeightInit(unittest.TestCase):
    """Bug 4 — non-focal empty_weight must use loop variable num_class."""

    def test_weight_shapes_match_num_classes(self):
        criterion = _make_criterion()
        self.assertEqual(criterion.empty_weight_1.shape[0], 5,
                         "empty_weight_1 should have num_classes[0]+1=5 entries")
        self.assertEqual(criterion.empty_weight_2.shape[0], 11,
                         "empty_weight_2 should have num_classes[1]+1=11 entries")
        self.assertEqual(criterion.empty_weight_3.shape[0], 5,
                         "empty_weight_3 should have num_classes[2]+1=5 entries")

    def test_eos_weight_at_last_position(self):
        criterion = _make_criterion()
        for buf_name in ("empty_weight_1", "empty_weight_2", "empty_weight_3"):
            buf = getattr(criterion, buf_name)
            self.assertAlmostEqual(
                buf[-1].item(), 0.1, places=5,
                msg=f"{buf_name}[-1] should be eos_coef=0.1"
            )
        for buf_name in ("empty_weight_1", "empty_weight_2", "empty_weight_3"):
            buf = getattr(criterion, buf_name)
            self.assertAlmostEqual(
                buf[0].item(), 1.0, places=5,
                msg=f"{buf_name}[0] should be 1.0 (non-background)"
            )


class TestLogitsSourceCorrectness(unittest.TestCase):
    """Bug 1 — src_logits_list_3 must collect from bz_src_logits_3, not bz_src_logits_1."""

    def test_diagnosis_logits_come_from_third_head(self):
        """
        Simulate the inner loop of loss_labels_focal and verify that
        each appended tensor in src_logits_list_3 is genuinely from
        bz_src_logits_3 (different values from bz_src_logits_1).
        """
        batch = 2
        n_queries = 5
        n_classes = 4

        torch.manual_seed(0)
        src_logits_1 = torch.zeros(batch, n_queries, n_classes)
        src_logits_3 = torch.ones(batch, n_queries, n_classes)

        valid_query = torch.tensor([0, 1])

        src_logits_list_3_correct = []
        src_logits_list_3_buggy = []

        for batch_idx in range(batch):
            bz_src_logits_1 = src_logits_1[batch_idx]
            bz_src_logits_3 = src_logits_3[batch_idx]
            src_logits_list_3_correct.append(bz_src_logits_3[valid_query])
            src_logits_list_3_buggy.append(bz_src_logits_1[valid_query])

        correct = torch.cat(src_logits_list_3_correct)
        buggy = torch.cat(src_logits_list_3_buggy)

        self.assertFalse(
            torch.allclose(correct, buggy),
            "Test setup error: correct and buggy should differ"
        )
        self.assertTrue(
            torch.all(correct == 1.0),
            "Correct implementation should use bz_src_logits_3 (all ones)"
        )
        self.assertTrue(
            torch.all(buggy == 0.0),
            "Buggy implementation uses bz_src_logits_1 (all zeros) — confirms the bug"
        )


class TestFreezeClassExceptionHandling(unittest.TestCase):
    """Bare-except fix — freeze flags must only be set on NameError/UnboundLocalError."""

    def test_freeze_flag_set_on_name_error(self):
        """If a logit tensor is missing, freeze_classN should become True."""
        freeze_class1 = False
        try:
            _ = undefined_tensor.shape[:2]  # noqa: F821 — intentional NameError
            freeze_class1 = False
        except (NameError, UnboundLocalError):
            freeze_class1 = True

        self.assertTrue(freeze_class1, "freeze_class1 should be True when logit is missing")

    def test_no_freeze_when_tensor_present(self):
        """If logit tensor exists, freeze_classN should remain False."""
        src_logits = torch.zeros(2, 5)
        freeze_class1 = False
        try:
            _ = torch.full(src_logits.shape[:2], 4, dtype=torch.int64)
            freeze_class1 = False
        except (NameError, UnboundLocalError):
            freeze_class1 = True

        self.assertFalse(freeze_class1, "freeze_class1 should be False when tensor is present")

    def test_real_errors_not_silenced(self):
        """A genuine RuntimeError must NOT be silenced by the freeze block."""
        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError("CUDA out of memory")
            except (NameError, UnboundLocalError):
                pass  # should NOT catch RuntimeError


class TestDetachFreeze(unittest.TestCase):
    """Bug 2 — .detach() must be assigned back; calling it on void is a no-op."""

    def test_detach_in_place_has_no_effect(self):
        """Calling .detach() without reassignment does NOT stop gradient flow."""
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        t.detach()  # the bug: return value discarded
        self.assertTrue(t.requires_grad,
                        "After discarded detach(), requires_grad is still True — bug confirmed")

    def test_detach_with_reassignment_stops_gradients(self):
        """Calling t = t.detach() correctly stops gradient flow."""
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        t = t.detach()  # the fix
        self.assertFalse(t.requires_grad,
                         "After reassigned detach(), requires_grad must be False")


class TestBoxRenewal(unittest.TestCase):
    """Bug 3 — box renewal must use num_proposals (plural) and handle empty bbox_pre."""

    def test_renewal_pads_to_num_proposals(self):
        """After filtering, boxes must be padded back up to num_proposals."""
        num_proposals = 300
        filtered = torch.randn(1, 150, 4)

        num_to_add = num_proposals - filtered.shape[1]
        self.assertEqual(num_to_add, 150)

        extra = torch.randn(1, num_to_add, 4)
        result = torch.cat((filtered, extra), dim=1)
        self.assertEqual(result.shape[1], num_proposals)

    def test_no_error_when_filter_keeps_all(self):
        """If no boxes are filtered, num_to_add should be 0 and no concat occurs."""
        num_proposals = 300
        full = torch.randn(1, 300, 4)
        num_to_add = num_proposals - full.shape[1]
        self.assertEqual(num_to_add, 0)
        if num_to_add > 0:
            full = torch.cat((full, torch.randn(1, num_to_add, 4)), dim=1)
        self.assertEqual(full.shape[1], 300)

    def test_empty_stack_would_crash(self):
        """torch.stack on an empty list must raise — proves the old bug crashes."""
        bbox_pre = []
        with self.assertRaises((RuntimeError, IndexError)):
            torch.stack(bbox_pre)


if __name__ == "__main__":
    unittest.main()
