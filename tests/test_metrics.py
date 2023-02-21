from rl_tools.metrics import Metrics


def test_metrics():
    metrics = Metrics("a", "b", "c")

    metrics.reset()
    assert len(metrics["a"]) == 0

    metrics.add("a", 1)
    metrics.add("b", 2)

    assert metrics["a"][0] == 1
    assert metrics["b"][0] == 2
    assert len(metrics["c"]) == 0

    metrics.reset()
    assert len(metrics["a"]) == 0
