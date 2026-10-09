"""The steps of an episode against the paper's runs: for three instances, the beta and stopping tolerance of
every step and the sigma and nu of every perturbation, as the code of the paper's runs produced them
(episode_reference.json). The L-BFGS-B calls and the perturbation are replaced by stand-ins, so only the
sequence of steps is run; the starting configuration can then be any array."""
import json
import types
from pathlib import Path

import sphcode  # noqa: F401  (sets single-threaded BLAS; import it before numpy)
import numpy as np
import pytest
from sphcode import episode
from sphcode.sources import resolve

REFERENCE = json.loads((Path(__file__).parent / "episode_reference.json").read_text())


@pytest.mark.parametrize("ref", REFERENCE, ids=lambda r: f"{r['n']}x{r['d']}")
def test_steps_follow_the_papers_runs(ref, monkeypatch):
    monkeypatch.setattr(episode, "minimize", lambda fun, x0, **kw: types.SimpleNamespace(x=x0))
    monkeypatch.setattr(episode, "perturb", lambda pts, sigma, nu, rng: pts)
    n, d = ref["n"], ref["d"]
    events = []

    def log(i, event, cos, best_cos):
        events.append(["perturb", event["sigma"], event["nu"]] if event["perturb"]
                      else ["step", event["beta"], event["eps"]])

    u, _ = resolve("per-instance", n, d)
    episode.run_episode(n, d, u, time_limit=1e9, seed=1, start=np.eye(n, d) + 1.0, max_steps=60, log=log)
    assert [e[0] for e in events] == [e[0] for e in ref["events"]]
    assert np.array([e[1:] for e in events]) == pytest.approx(np.array([e[1:] for e in ref["events"]]), rel=1e-12)
