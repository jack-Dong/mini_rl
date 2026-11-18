"""CPU regression for zero-advantage regularization and final evaluation."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch, NumPy and matplotlib are required")
class TrainingRegressionTests(unittest.TestCase):
    def test_zero_advantage_still_optimizes_entropy_and_evaluates_after_update(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
        transformers.AutoTokenizer = type("AutoTokenizer", (), {})
        vllm = types.ModuleType("vllm")
        vllm.LLM = type("LLM", (), {})
        vllm.SamplingParams = type("SamplingParams", (), {})

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.0, 1.0]))

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(logits=self.weight.expand(*input_ids.shape, 2))

        model = Model()
        initial = model.weight.detach().clone()
        args = SimpleNamespace(tis_max=2.0, group_size=2, seed=0, model_id="mock",
                               steps=1, staleness=0, eval_every=1, sample_steps=[],
                               num_prompts=1, reward_mode="group_centered", micro_batch_size=2,
                               max_grad_norm=10.0, max_new_tokens=1)
        rollout = {
            "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
            "prompt_mask": torch.ones((2, 1), dtype=torch.long),
            "gen_tokens": torch.zeros((2, 1), dtype=torch.long),
            "loss_mask": torch.ones((2, 1), dtype=torch.bool),
            "samp_token_logp": torch.zeros((2, 1)),
            "samp_topk_logp": torch.zeros((2, 1, 1)),
            "topk_ids": torch.zeros((2, 1, 1), dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"transformers": transformers, "vllm": vllm}):
            from src import vllm_suite as suite
            state_path = Path(directory) / "training.pt"
            with patch.object(suite.AutoModelForCausalLM, "from_pretrained", return_value=model, create=True), \
                 patch.object(suite, "build_sampler_weights", return_value=[]), \
                 patch.object(suite, "load_weights_into_vllm"), \
                 patch.object(suite, "_sample_rl_batch", return_value=(rollout, [0., 0.], [0., 0.], 0., 1)), \
                 patch.object(suite, "_build_optimizer_and_scheduler", side_effect=lambda m, v, a: (torch.optim.SGD(m.parameters(), lr=0.1), None)), \
                 patch.object(suite, "vllm_evaluate_policy", side_effect=lambda llm, m, *a, **kw: (float(m.weight[0].detach()), {"success": [], "failure": []})):
                result = suite.run_single_vllm_method(
                    "pg", {"weight": initial}, None, torch.float32, None, [], [], [], args,
                    torch.device("cpu"), False, 0., "test", init_eval=(0., {}),
                    variant=suite.VariantConfig(key="test", label="test", base_method="pg", entropy_coef=1.0),
                    training_state_path=state_path, experiment_id="correct",
                )
                trained = model.weight.detach().clone()
                model.weight.data.copy_(initial)
                resumed = suite.run_single_vllm_method(
                    "pg", {"weight": initial}, None, torch.float32, None, [], [], [], args,
                    torch.device("cpu"), False, 0., "test", init_eval=(0., {}),
                    variant=suite.VariantConfig(key="test", label="test", base_method="pg", entropy_coef=1.0),
                    training_state_path=state_path, experiment_id="correct", resume_training=True,
                )
                self.assertTrue(torch.equal(trained, model.weight.detach()))
                self.assertEqual(resumed["history"][-1]["eval_step"], 1.0)
                with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                    suite.run_single_vllm_method(
                        "pg", {"weight": initial}, None, torch.float32, None, [], [], [], args,
                        torch.device("cpu"), False, 0., "test", init_eval=(0., {}),
                        variant=suite.VariantConfig(key="test", label="test", base_method="pg", entropy_coef=1.0),
                        training_state_path=state_path, experiment_id="wrong", resume_training=True,
                    )
                model.weight.data.copy_(trained)
        self.assertFalse(torch.equal(initial, model.weight.detach()))
        self.assertEqual(result["history"][-1]["eval_step"], 1.0)
        self.assertAlmostEqual(result["history"][-1]["eval_reward"], float(model.weight[0].detach()))
        self.assertTrue(result["complete"])


if __name__ == "__main__":
    unittest.main()
