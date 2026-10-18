"""Verify scheduled optimization retains its trajectory across checkpoint resume."""

import copy

import pytest

torch = pytest.importorskip("torch")

from solar_filament.learning_rate import learning_rate_at_step, validate_learning_rate_schedule  # noqa: E402
from solar_filament.training_iterator import training_iterator  # noqa: E402

COSINE = {"learning_rate": 1e-4, "final_learning_rate": 1e-5, "learning_rate_policy": "cosine", "epochs": 28, "steps_per_epoch": 160}


def test_schedule_follows_absolute_horizon_and_old_configs_keep_constant_lr():
    rates = [learning_rate_at_step(COSINE, step) for step in range(4480)]
    assert rates[0] == 1e-4
    assert rates[-1] == 1e-5
    assert all(left >= right for left, right in zip(rates, rates[1:]))
    assert learning_rate_at_step({"learning_rate": 1e-4}, 100000) == 1e-4
    assert learning_rate_at_step(dict(COSINE, epochs=1, steps_per_epoch=1), 0) == 1e-4


@pytest.mark.parametrize("change", [{"learning_rate_policy": "unknown"}, {"final_learning_rate": 0}, {"final_learning_rate": 2e-4}, {"final_learning_rate": float("nan")}, {"epochs": 0}, {"epochs": True}, {"steps_per_epoch": 1.5}])
def test_invalid_policy_rejected_before_training(change):
    with pytest.raises(ValueError):
        validate_learning_rate_schedule(dict(COSINE, **change))


@pytest.mark.parametrize("step", [-1, 4480, True, 1.5])
def test_invalid_training_position_rejected(step):
    with pytest.raises(ValueError):
        learning_rate_at_step(COSINE, step)


@pytest.mark.parametrize("pause_after", [5, 12])
def test_resumed_adamw_parameters_and_moments_equal_uninterrupted_training(pause_after):
    config = dict(COSINE, epochs=3, steps_per_epoch=7)
    torch.manual_seed(712)
    initial_rng = torch.get_rng_state().clone()

    def make_state():
        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.8], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"], weight_decay=1e-4)
        return parameter, optimizer

    def advance(parameter, optimizer, start, stop):
        for step in range(start, stop):
            optimizer.param_groups[0]["lr"] = learning_rate_at_step(config, step)
            target = torch.randn(2, dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            (parameter - target).square().sum().backward()
            optimizer.step()

    complete, full_optimizer = make_state()
    torch.set_rng_state(initial_rng)
    advance(complete, full_optimizer, 0, 21)
    final_rng = torch.get_rng_state().clone()

    interrupted, optimizer = make_state()
    torch.set_rng_state(initial_rng)
    advance(interrupted, optimizer, 0, pause_after)
    checkpoint = {"parameter": interrupted.detach().clone(), "optimizer": copy.deepcopy(optimizer.state_dict()), "epoch": pause_after // 7, "step": pause_after % 7, "rng": torch.get_rng_state().clone()}
    resumed, resumed_optimizer = make_state()
    with torch.no_grad():
        resumed.copy_(checkpoint["parameter"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    start = checkpoint["epoch"] * config["steps_per_epoch"] + checkpoint["step"]
    advance(resumed, resumed_optimizer, start, 21)
    assert torch.equal(resumed, complete)
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert resumed_optimizer.param_groups[0]["lr"] == full_optimizer.param_groups[0]["lr"]
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(resumed_optimizer.state[resumed][key], full_optimizer.state[complete][key])


@pytest.mark.parametrize("pause_after", [5, 7])
def test_dataloader_checkpoint_replay_does_not_draw_an_extra_global_seed(pause_after):
    config = dict(COSINE, epochs=3, steps_per_epoch=7)
    loader = torch.utils.data.DataLoader(torch.arange(7), batch_size=1, shuffle=False, num_workers=0)

    def train(resume_state=None):
        torch.manual_seed(231)
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        start_epoch = skip = 0
        if resume_state:
            with torch.no_grad():
                parameter.copy_(resume_state["parameter"])
            optimizer.load_state_dict(resume_state["optimizer"])
            torch.set_rng_state(resume_state["rng"])
            start_epoch, skip = resume_state["epoch"], resume_state["step"]
        saved = None
        for epoch in range(start_epoch, 3):
            iterator = training_iterator(loader, continuing_epoch=epoch == start_epoch and skip > 0)
            for step, _ in enumerate(iterator):
                if epoch == start_epoch and step < skip:
                    continue
                absolute = epoch * 7 + step
                optimizer.param_groups[0]["lr"] = learning_rate_at_step(config, absolute)
                optimizer.zero_grad(set_to_none=True)
                (parameter - torch.randn(2, dtype=torch.float64)).square().sum().backward()
                optimizer.step()
                if resume_state is None and absolute + 1 == pause_after:
                    saved = {"parameter": parameter.detach().clone(), "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": torch.get_rng_state().clone(), "epoch": epoch, "step": step + 1}
            skip = 0
        return parameter, optimizer, torch.get_rng_state(), saved

    uninterrupted, optimizer, rng, saved = train()
    resumed, resumed_optimizer, resumed_rng, _ = train(saved)
    assert torch.equal(uninterrupted, resumed)
    assert torch.equal(rng, resumed_rng)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizer.state[uninterrupted][key], resumed_optimizer.state[resumed][key])
