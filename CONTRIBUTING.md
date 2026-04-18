# Contributing to kaggle-ml

Thanks for considering a contribution! Here's how to get involved.

## Adding a New Competition Domain

1. Create a new reference file in `references/` (e.g., `robotics.md`)
2. Follow the structure of existing reference files — include winning stack, key techniques, code templates, and common pitfalls
3. Add the domain to the competition type table in `SKILL.md`
4. Add the filename to `required_refs` in `scripts/validate.py`
5. Run `python scripts/validate.py` to confirm everything passes

## Updating Research Findings

If a new paper or competition result changes the meta (e.g., a new architecture beats current SOTA), update the relevant reference file and the "2026 Research Breakthroughs" section in `SKILL.md`.

## Code Quality

- All Python code in reference files should be runnable snippets (not pseudocode)
- Pin library versions where it matters (e.g., `elasticsearch<9`)
- Include seed-pinning and reproducibility in every pipeline

## Submitting Changes

1. Fork the repo
2. Create a feature branch (`git checkout -b add-robotics-domain`)
3. Make your changes
4. Run `python scripts/validate.py` — must pass with ✅
5. Open a PR with a clear description of what you added/changed

## Rebuilding the .skill File

After making changes, repackage the skill:

```bash
cd kaggle-ml-skill
zip -r kaggle-ml.skill kaggle-ml/ -x "*.git*"
```

(Adjust paths as needed for your setup.)

## Questions?

Open an issue — happy to discuss before you start working on something.
