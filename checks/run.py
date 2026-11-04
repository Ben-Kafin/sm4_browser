"""
The runner of the package's checks: every check module in a fixed order, each its run() (the
number of failures; one [ok] / [FAIL] line per sub-check), one line per module,
    <name>: ok (<n> checks, <t> s)   or   <name>: FAIL (<failures>)   or   <name>: not run (<why>),
and exit status 1 when any failed or could not run (a check that cannot run in the environment is
reported as not run and counts as a failure of the runner, never as a pass).  No timeout, no
parallelism, no retries: a check is small because it does one thing on synthetic data.

    python -m sm4_Browser.checks.run [names]

from the parent directory of a folder named sm4_Browser; with names given, only those modules run.
The checks need no .sm4 file (checks/synth.py experiments or numpy arrays); window_smoke drives the
windows on a virtual screen.  Scripts that need a real file or an open window are tools
(sm4_Browser/tools/), run by hand, not here.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import importlib
import io
import sys
import time
import traceback

# The order the checks run in (the table of the stage-2 document): the pipeline first, then the
# calibration's pin rule, the numpy-only geometry checks, the stripe matching, the sweeps, the windows.
CHECKS = ('pipeline_check', 'pin_rule_check', 'contour_check', 'pass_average_check', 'affine_check', 'stripe_check',
          'sweeps_check', 'window_smoke')


class _Tee(io.TextIOBase):
    """Stdout written through, and counted: the number of '[ok]' lines a check printed."""

    def __init__(self, out):
        self.out = out
        self.n_ok = 0

    def write(self, text):
        self.n_ok += text.count('[ok]')
        return self.out.write(text)

    def flush(self):
        self.out.flush()


def run(names=None):
    """Run the checks (all, or those named); returns the number of modules that failed or could not run."""
    selected = [n for n in CHECKS if not names or n in names]
    unknown = [n for n in (names or []) if n not in CHECKS]
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}; known: {', '.join(CHECKS)}")
    bad = 0
    for name in selected:
        print(f"== {name} ==")
        tee = _Tee(sys.stdout)
        old = sys.stdout
        sys.stdout = tee
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f"sm4_Browser.checks.{name}")
            failures = mod.run()
            why = None
        except Exception as exc:
            failures, why = None, f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            sys.stdout = old
        dt = time.perf_counter() - t0
        if why is not None:
            print(f"{name}: not run ({why})")
            bad += 1
        elif isinstance(failures, str):
            print(f"{name}: not run ({failures})")
            bad += 1
        elif failures:
            print(f"{name}: FAIL ({failures})")
            bad += 1
        else:
            print(f"{name}: ok ({tee.n_ok} checks, {dt:.1f} s)")
    return bad + len(unknown)


if __name__ == '__main__':
    sys.exit(1 if run(sys.argv[1:]) else 0)
