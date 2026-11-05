# checks: the package's one set of checks

**To any future conversation or session working on this package (people and AI assistants alike):
verify your work with the checks in this folder, and keep them up to date. Do not write your own
test or smoke scripts somewhere else.**

1. **Run these checks** to see whether a change works and whether it broke anything:

   ```
   PYTHONDONTWRITEBYTECODE=1 python -m sm4_Browser.checks.run [names]
   ```

   from the parent directory of the package folder (named `sm4_Browser`). With names given, only
   those checks run (e.g. `... checks.run save_check`).
2. **Edit the checks here, never a private copy.** When a change needs a new test, extend the check
   module that covers that feature, or add a new `<feature>_check.py` module here. Do not leave a
   one-off script in a scratch folder.
3. **Register a new check.** Add its name to `CHECKS` in `checks/run.py` and give it an entry in
   the checks list of the top-level `README.md` (section "The checks").
4. **Keep the checks true to the code.** When a change alters behaviour that a check covers, update
   that check in the same piece of work, so the checks always describe the code as it is.
5. **Commit and push check changes to `main`** together with the change they belong to (Benjamin's
   rule, 2026-11-05).
6. **Never weaken a check to make it pass.** Do not remove, loosen or skip a check because the code
   fails it. A failing check means the code or its input is wrong: find out why and fix that.

## Conventions of a check module

- `run()` returns the number of failures (0 = pass), or a string saying why the check could not run
  (the runner counts that as a failure, never as a pass). It prints one `[ok]` / `[FAIL]` line per
  sub-check (the runner counts the `[ok]` lines).
- It runs on synthetic data (`checks/synth.py` experiments, or numpy arrays): no real `.sm4` file,
  and nothing written outside a temporary folder it removes.
- It can run on its own: `python -m sm4_Browser.checks.<name>`, or F5 in Spyder.
- Windows draw on a real screen, never on Qt's offscreen platform (it errs in the browser and freezes
  the processing): the display at hand, or an Xvfb virtual screen started by
  `window_smoke._virtual_screen` (reuse it; do not write another). A check that opens windows after
  another check's screen has stopped runs its pass in a child process with its own screen, as
  `save_check` does.
- Scripts that need a real file or an open window to look at are tools (`tools/`), run by hand, not
  checks.

## The checks (details in the top-level README, section "The checks")

| module | what it checks |
|---|---|
| `pipeline_check` | the one pipeline path (`sts.STS.stages` / `STS.view`) on synthetic sweeps |
| `pin_rule_check` | the calibration locks' pin rule |
| `contour_check` | the contour slice's marching squares and placement |
| `pass_average_check` | the average of two passes |
| `affine_check` | the affine drift transform |
| `stripe_check` | the stripe matching against known truth |
| `sweeps_check` | several sweeps per location: layouts, pairs, views, clock |
| `window_smoke` | the windows on one synthetic experiment: state, drawn arrays, records |
| `save_check` | the saving of every window: exactly what it shows plus its full info panel |

`synth.py` builds the synthetic experiments; `run.py` is the runner.

## Running them in a fresh Linux container

The package's requirements (top-level `README.md`) plus, for the window checks, an X server and the
libraries Qt's xcb platform loads. In an Ubuntu container on 2026-11-05 these were enough:

```
pip install "PySide6<6.12" pyqtgraph numpy scipy spym xarray hvplot "opencv-python-headless<5" matplotlib loky
apt-get install -y xvfb libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-shape0 libxcb-xinerama0 libxcb-xkb1
```
