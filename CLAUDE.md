# Instructions for every session in this repository

## Checks and tests: use the ones in `checks/`

Read `checks/README.md` before verifying any change. In short:

- Verify every change with the package's own checks: from the parent directory of the package
  folder (named `sm4_Browser`), `PYTHONDONTWRITEBYTECODE=1 python -m sm4_Browser.checks.run [names]`.
- Do not write your own test or smoke scripts anywhere else. When a change needs a new test,
  extend the check module in `checks/` that covers the feature, or add a new `<feature>_check.py`
  there, register it in `CHECKS` in `checks/run.py` and in the checks list of the top-level
  `README.md` (section "The checks").
- When a change alters behaviour a check covers, update that check in the same piece of work.
- Commit and push check changes to `main` together with the change they belong to.
- Never remove, loosen or skip a check to make it pass: a failing check means the code or its input
  is wrong.
