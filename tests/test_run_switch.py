"""The CacheScout on/off switch and ablation resolution (no GPU)."""

import pytest

from cachescout.run import resolve_system
from cachescout.sim.simulator import system_flags


@pytest.mark.parametrize(
    ("exp", "cli", "expected"),
    [
        ({}, None, "vanilla"),
        ({"cachescout": {"enabled": False}}, None, "vanilla"),
        ({"cachescout": {"enabled": True}}, None, "cachescout"),
        ({"cachescout": {"enabled": True, "warmup": False}}, None, "eviction_only"),
        ({"cachescout": {"enabled": True, "eviction": False}}, None, "warmup_only"),
        ({"cachescout": {"enabled": True}}, "vanilla", "vanilla"),      # CLI wins
        ({"cachescout": {"enabled": False}}, "cachescout", "cachescout"),
        ({"system": "continuum", "cachescout": {"enabled": True}}, None, "continuum"),
    ],
)
def test_resolve_system(exp: dict, cli: str | None, expected: str) -> None:
    assert resolve_system(exp, cli) == expected


def test_resolve_rejects_unknown_and_empty_ablation() -> None:
    with pytest.raises(ValueError):
        resolve_system({}, "bogus")
    with pytest.raises(ValueError):
        resolve_system({"cachescout": {"enabled": True, "eviction": False, "warmup": False}}, None)


def test_system_flags() -> None:
    assert system_flags("vanilla") == (None, False)          # stock vLLM scheduler
    assert system_flags("eviction_only") == ("cachescout", False)
    assert system_flags("warmup_only") == ("lru", True)
    assert system_flags("cachescout") == ("cachescout", True)
    assert system_flags("continuum") == ("continuum", False)


def test_apply_variant_overrides_params() -> None:
    from cachescout.run import apply_variant

    exp = {"cachescout": {"params": {"tau": 0.1, "scope": "session"}},
           "variants": {"literal": {"system": "eviction_only",
                                    "params": {"scope": "global", "block_mapping": "all"}}}}
    assert apply_variant(exp, "literal") == "eviction_only"
    assert exp["cachescout"]["params"] == {"tau": 0.1, "scope": "global", "block_mapping": "all"}
    with pytest.raises(ValueError):
        apply_variant(exp, "missing")
