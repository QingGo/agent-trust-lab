"""Generate 3-5 trap variants per high-discrimination type for benchmark expansion.

Targets the 8 types from Stage 2.3 analysis that have highest inter-model discrimination.
Uses RedTeamGenerator with domain/context/tool swaps and LLM refinement.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent_trust_lab.traps
from agent_trust_lab.redteam.generator import RedTeamConfig, RedTeamGenerator

_HERE = Path(__file__).resolve().parent.parent
_TRAP_DATA_DIR = str(Path(agent_trust_lab.traps.__file__).parent / "data")

HIGH_DISCRIM_TYPES = [
    "human_interaction_spoof",
    "loop_induction",
    "memory_pollution",
    "multi_turn_gradual_pollution",
    "parameter_hallucination",
    "tool_bypass",
    "tool_description_poisoning",
    "tool_parameter_coercion",
]


def main():
    parser = argparse.ArgumentParser(description="Generate trap variants for benchmark expansion")
    parser.add_argument("--variants", type=int, default=4, help="Variants per source trap")
    parser.add_argument("--output", default="results/variants", help="Output directory for YAML files")
    parser.add_argument("--llm-refine", action="store_true", help="Use LLM refinement (needs API key)")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model for LLM refinement")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = RedTeamConfig(
        trap_library_path=_TRAP_DATA_DIR,
        output_dir=args.output,
        num_variants=args.variants,
        domain_swap=True,
        context_swap=True,
        tool_swap=True,
        severity_vary=True,
        difficulty_vary=True,
        mutation_seed=args.seed,
        llm_refine=args.llm_refine,
        llm_model=args.model,
        target_types=HIGH_DISCRIM_TYPES,
    )

    os.makedirs(args.output, exist_ok=True)

    generator = RedTeamGenerator(config)
    candidates = generator.generate()

    print(f"\nGenerated {len(candidates)} variants in {args.output}/")
    print(f"Target types: {len(HIGH_DISCRIM_TYPES)}")
    print(f"Variants per source: {args.variants}")
    print(f"Expected: {len(HIGH_DISCRIM_TYPES) * args.variants} files")


if __name__ == "__main__":
    main()
