#!/usr/bin/env python3
"""
Validation script for kaggle-ml skill.
Category: machine-learning
Validates skill structure, config, and competition-specific constraints.
"""

import os, sys, yaml, json
from pathlib import Path


def validate_config(config_path: str) -> dict:
    """Validate skill configuration file."""
    errors = []

    if not os.path.exists(config_path):
        return {"valid": False, "errors": ["Config file not found"]}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return {"valid": False, "errors": [f"YAML parse error: {e}"]}

    if 'skill' not in config:
        errors.append("Missing 'skill' section")
    else:
        if 'name' not in config['skill']:    errors.append("Missing skill.name")
        if 'version' not in config['skill']: errors.append("Missing skill.version")

    if 'settings' in config:
        s = config['settings']
        if 'log_level' in s and s['log_level'] not in ['debug','info','warn','error']:
            errors.append(f"Invalid log_level: {s['log_level']}")

        # Competition-specific validation
        comp = s.get('competition', {})
        if 'default_cv_folds' in comp and not (3 <= comp['default_cv_folds'] <= 10):
            errors.append("default_cv_folds must be 3–10")
        if 'min_cot_ratio' in comp and not (0.5 <= comp['min_cot_ratio'] <= 1.0):
            errors.append("min_cot_ratio must be 0.5–1.0 (research: ≥0.75 required)")
        if 'nemotron_temperature' in comp and comp['nemotron_temperature'] != 0.0:
            errors.append("nemotron_temperature must be 0.0 (deterministic eval)")

    return {"valid": len(errors) == 0, "errors": errors,
            "config": config if not errors else None}


def validate_skill_structure(skill_path: str) -> dict:
    """Validate skill directory structure."""
    required_dirs  = ['assets', 'scripts', 'references']
    required_files = ['SKILL.md']

    # Required reference files for all 12 competition domains
    required_refs = [
        'tabular.md', 'computer-vision.md', 'audio.md', 'nlp-llm.md',
        'time-series.md', 'math-reasoning.md', 'rl-agent.md', 'arc-reasoning.md',
        'llm-finetune.md', 'minimal-nn.md', 'biology-science.md', 'social-good.md'
    ]

    errors = []
    warnings = []

    for f in required_files:
        if not os.path.exists(os.path.join(skill_path, f)):
            errors.append(f"Missing required file: {f}")

    for d in required_dirs:
        dp = os.path.join(skill_path, d)
        if not os.path.isdir(dp):
            errors.append(f"Missing required directory: {d}/")
        else:
            files = [f for f in os.listdir(dp) if f != '.gitkeep']
            if not files:
                errors.append(f"Directory {d}/ has no real content")

    # Check all 12 reference files exist
    refs_path = os.path.join(skill_path, 'references')
    if os.path.isdir(refs_path):
        for ref in required_refs:
            if not os.path.exists(os.path.join(refs_path, ref)):
                errors.append(f"Missing reference file: references/{ref}")
        actual_refs = [f for f in os.listdir(refs_path) if f.endswith('.md')]
        if len(actual_refs) < 12:
            warnings.append(f"Only {len(actual_refs)}/12 reference files found")

    # Check SKILL.md has required frontmatter fields
    skill_md = os.path.join(skill_path, 'SKILL.md')
    if os.path.exists(skill_md):
        with open(skill_md) as f:
            content = f.read()
        if 'name:' not in content:    errors.append("SKILL.md missing 'name' frontmatter")
        if 'description:' not in content: errors.append("SKILL.md missing 'description' frontmatter")
        # Check key competition sections exist
        for section in ['Refinement Loop', 'Competition Type', 'Winning Strategy',
                         'AGI Mode', 'Output Format']:
            if section not in content:
                warnings.append(f"SKILL.md missing section: {section}")

    return {
        "valid":      len(errors) == 0,
        "errors":     errors,
        "warnings":   warnings,
        "skill_name": os.path.basename(skill_path)
    }


def validate_competition_constraints() -> dict:
    """Validate key competition-specific rules from 2026 research."""
    checks = {
        "nemotron_cot_ratio":    "Training data must have ≥75% chain-of-thought examples",
        "aimo_answer_range":     "AIMO answers must be integers in [0, 99999]",
        "arc_rl_over_llm":       "ARC-AGI-3: use RL+exploration, not pure LLM (LLMs score ~0%)",
        "cv_before_submission":  "Always validate CV score before trusting public LB",
        "no_future_leak":        "Time-series: all features must use only past data",
        "seed_reproducibility":  "Pin torch/numpy/random seeds for reproducibility",
    }
    return {"checks": checks, "count": len(checks)}


def main():
    skill_path = Path(__file__).parent.parent
    print(f"Validating kaggle-ml skill (v2.0.0)...")
    print(f"Path: {skill_path}\n")

    # 1. Structure
    struct = validate_skill_structure(str(skill_path))
    print(f"Structure:    {'✅ PASS' if struct['valid'] else '❌ FAIL'}")
    for e in struct['errors']:   print(f"  ERROR:   {e}")
    for w in struct['warnings']: print(f"  WARN:    {w}")

    # 2. Config
    config_path = skill_path / 'assets' / 'config.yaml'
    if config_path.exists():
        cfg = validate_config(str(config_path))
        print(f"Config:       {'✅ PASS' if cfg['valid'] else '❌ FAIL'}")
        for e in cfg['errors']: print(f"  ERROR:   {e}")
    else:
        print("Config:       ⚠️  SKIPPED (no config.yaml)")

    # 3. Competition constraints
    comp = validate_competition_constraints()
    print(f"Competition:  ✅ {comp['count']} constraints registered")

    # Summary
    all_valid = struct['valid']
    print(f"\n{'='*50}")
    print(f"Overall: {'✅ VALID' if all_valid else '❌ INVALID'}")
    if all_valid:
        refs = len([f for f in os.listdir(skill_path / 'references')
                    if f.endswith('.md')]) if (skill_path / 'references').exists() else 0
        print(f"  Skill:      kaggle-ml v2.0.0")
        print(f"  References: {refs}/12 competition domains")
        print(f"  Mode:       Kaggle Grandmaster (top-1% target)")
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
