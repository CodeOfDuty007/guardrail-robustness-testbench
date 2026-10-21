"""Attack families (ATTACK_FAMILIES_AND_DEFENSE_PLAN.md Part A + D). See base.py for the contract."""
from app.engine.families.base import (  # noqa: F401
    FAMILY_REGISTRY, AttackFamily, CandidatePrompt, Turn, ask_attacker, family_rng, get_family, list_families,
    load_all, register_family,
)
