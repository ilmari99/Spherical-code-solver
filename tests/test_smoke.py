import sphcode  # noqa: F401  (sets single-threaded BLAS; import it before numpy)
import numpy as np
from sphcode.cli import main
from sphcode.episode import run_episode
from sphcode.sources import resolve


def test_solve_small_instance():
    u, _ = resolve("global", 30, 3)
    r = run_episode(30, 3, u, time_limit=1.0, seed=1)
    pts = r["points"]
    assert pts.shape == (30, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert r["best_ang"] >= r["start_ang"] >= r["input_ang"] - 1e-9
    assert r["steps"] > 0


def test_step_capped_episode_is_reproducible():
    u, _ = resolve("per-instance", 40, 4)
    a = run_episode(40, 4, u, time_limit=1e9, seed=5, max_steps=12)
    b = run_episode(40, 4, u, time_limit=1e9, seed=5, max_steps=12)
    assert a["best_cos"] == b["best_cos"]
    assert np.array_equal(a["points"], b["points"])


def test_supplied_configuration_with_predicted_parameters():
    rng = np.random.default_rng(0)
    start = rng.normal(size=(25, 4))
    u, _ = resolve("predictor", 25, 4)
    r = run_episode(25, 4, u, time_limit=0.5, seed=2, start=start)
    assert r["points"].shape == (25, 4)
    assert r["best_ang"] >= r["input_ang"]


def test_supplied_configuration_is_never_made_worse():
    u, _ = resolve("global", 30, 3)
    first = run_episode(30, 3, u, time_limit=1e9, seed=1, max_steps=40)
    again = run_episode(30, 3, u, time_limit=1e9, seed=1, max_steps=2, start=first["points"])
    assert again["best_cos"] <= first["best_cos"] + 1e-15


def test_text_output_is_comma_separated_and_reads_back_as_a_start(tmp_path):
    out = str(tmp_path / "points.txt")
    main(["solve", "30", "3", "--params", "global", "--time-limit", "0.5", "--seed", "1", "--out", out])
    assert np.loadtxt(out, delimiter=",").shape == (30, 3)
    main(["solve", "30", "3", "--params", "global", "--time-limit", "0.5", "--seed", "1", "--start", out])
