import torch

from crossfit.causal_patching import error_preserving_fit_patch, standardized_patch_effect
from crossfit.crosscoder import CrossCoder


def test_no_ablation_is_exact_error_preserving_splice():
    torch.manual_seed(2)
    model = CrossCoder(4, 1, 2, decoder_bias=False).eval()
    evo = torch.randn(5, 4)
    fit = torch.randn(5, 4)
    patched, _ = error_preserving_fit_patch(model, evo, fit, ablated_feature=None)
    assert torch.allclose(patched, fit, atol=2e-7, rtol=0)


def test_dead_feature_ablation_has_no_effect():
    torch.manual_seed(3)
    model = CrossCoder(3, 1, 1, decoder_bias=False).eval()
    with torch.no_grad():
        model.encoder.weight[0].zero_()
        model.encoder.bias[0] = -1.0
    evo = torch.randn(6, 3)
    fit = torch.randn(6, 3)
    no_op, _ = error_preserving_fit_patch(model, evo, fit, ablated_feature=None)
    patched, code = error_preserving_fit_patch(model, evo, fit, ablated_feature=0)
    assert torch.count_nonzero(code[:, 0]) == 0
    assert torch.allclose(patched, no_op, atol=2e-7, rtol=0)


def test_standardized_patch_effect_uses_fixed_baseline_scale():
    output = standardized_patch_effect(torch.tensor([4.0, 1.0]), torch.tensor([2.0, 2.0]), 2.0)
    assert torch.equal(output, torch.tensor([1.0, -0.5]))
