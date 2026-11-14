# Adding a new evaluator

An evaluator judges one timeline point and says what, if anything, is wrong
with it. Everything else — finding the new points, assembling history, the
watermark, retries, committing, the log — belongs to the driver and is
identical for every algorithm. So installing a new evaluator is three small
code edits, one config file, and two dry runs before it is allowed to write.

The contract is in [`base.py`](base.py); read its module docstring first. The
two shipped evaluators are the worked examples: [`limits.py`](limits.py) with
no history, [`trends.py`](trends.py) with a lookback window.

## The checklist

1. Choose the `algo` name and the subject vocabulary.
2. Write `dqm/evaluate/<algo>.py` with one `Evaluator` subclass.
3. Register the class in `_registry()` in [`config.py`](config.py).
4. Export it from [`__init__.py`](__init__.py).
5. Write the driver YAML, in its own working directory.
6. Run `--check`, then report mode, and read the log.
7. Turn `commit` on and add the cron entry.

## 1. The names are permanent

An alarm episode is identified by `(algo, config, subject, sid)` — see
`ALARMS.key` in [`../schema.py`](../schema.py). Those strings go into the
database and the display reads them back generically, so nothing needs
teaching about a new algorithm; but equally, renaming a subject later does not
migrate anything. It orphans every episode already filed under the old name
and starts fresh ones. Pick the vocabulary once, keep it short, and keep it
lowercase with underscores (`low_stats`, not `"low stats"`), so a query for a
subject cannot silently miss rows.

`config` is not chosen here — it is a label in the YAML, which is how the same
algorithm can run twice over one source at two tightnesses without the two
colliding on alarm identity.

## 2. The module

The example below is a real gap in the current set: neither `limits` nor
`trends` reports a point whose measurements the producer itself flagged, since
both deliberately skip anything with a nonzero `code`. Call it `codes`.

```python
"""The `codes` evaluator: did the producer flag its own measurements?

`limits` and `trends` both skip a measurement whose code is nonzero -- a
missing value is not a physics finding, and reporting it as one would alarm on
the extractor's problem under a physics subject. That leaves nobody watching
the codes themselves, which is this algorithm's whole job.

    algo:   codes
    config: crv-codes-0
    sid:    115
    pars:
      level:    2          # severity to report, default alarm
      fraction: 0.0        # report only if this share of metrics is flagged
      metrics:  []         # empty means every metric in the point

Subjects:

    missing     at least one measurement came back missing or in error
    low_stats   at least one measurement was flagged as low statistics
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Sequence

from dqm.errors import DqmConfigError
from dqm.evaluate.base import LEVELS, Cause, Evaluator, Finding, Point

LOGGER = logging.getLogger("dqm.evaluate.codes")

__all__ = ["CodesEvaluator"]

# dqm.numbers.code is a fixed enum -- NumberCode in dqm/schema.py, mirroring
# DqmNumber::codeValues. Which subject each nonzero code belongs under.
SUBJECT_OF_CODE: Dict[int, str] = {1: "low_stats", 2: "missing", 3: "missing"}


class CodesEvaluator(Evaluator):
    """Report measurements the producer flagged as unusable."""

    algo = "codes"
    subjects = ("missing", "low_stats")
    lookback = 0

    def configure(self) -> None:
        # Validate pars here, at construction, so a bad config file fails in
        # the cron job's first second rather than an hour into a startup walk.
        level = self.pars.get("level", LEVELS["alarm"])
        if isinstance(level, str):
            if level not in LEVELS:
                raise DqmConfigError(
                    f"codes: level must be one of {sorted(LEVELS)} or 0-3, "
                    f"got {level!r}")
            level = LEVELS[level]
        if int(level) not in range(len(LEVELS)):
            raise DqmConfigError(f"codes: level must be 0-3, got {level!r}")
        self._level = int(level)

        try:
            self._fraction = float(self.pars.get("fraction", 0.0))
        except (TypeError, ValueError):
            raise DqmConfigError("codes: fraction must be a number")
        if not 0.0 <= self._fraction <= 1.0:
            raise DqmConfigError("codes: fraction must be between 0 and 1")

        metrics = self.pars.get("metrics") or []
        if not isinstance(metrics, (list, tuple)):
            raise DqmConfigError("codes: metrics must be a list")
        self._metrics = [str(m) for m in metrics]

    def _selected(self, point: Point) -> List[int]:
        if not self._metrics:
            return list(point.vids)
        return [vid for vid in point.vids
                if str(vid) in self._metrics or point.name(vid) in self._metrics]

    def evaluate(self, point: Point,
                 history: Sequence[Point] = ()) -> Sequence[Finding]:
        vids = self._selected(point)
        if not vids:
            return []

        causes: Dict[str, List[Cause]] = {s: [] for s in self.subjects}
        detail: Dict[str, List[dict]] = {s: [] for s in self.subjects}

        for vid in vids:
            m = point[vid]
            if m.ok:
                continue
            subject = SUBJECT_OF_CODE.get(m.code)
            if subject is None:          # an enum value this code predates
                continue
            causes[subject].append(point.cause(vid, level=self._level))
            detail[subject].append({"metric": point.name(vid), "vid": vid,
                                    "code": m.code})

        findings: List[Finding] = []
        for subject in self.subjects:
            if not causes[subject]:
                continue
            if len(causes[subject]) < self._fraction * len(vids):
                LOGGER.debug("%s: %d %s of %d metrics, below fraction %.2f",
                             point.label, len(causes[subject]), subject,
                             len(vids), self._fraction)
                continue
            LOGGER.info("%s: %d of %d metrics %s", point.label,
                        len(causes[subject]), len(vids), subject)
            # One finding per subject, every flagged metric a cause of it --
            # see rule (b) below.
            findings.append(self.finding(subject, *causes[subject],
                                         note=json.dumps({subject: detail[subject]})))
        return findings
```

### The rules `evaluate` has to keep

Four of them, and the first two are enforced — the driver calls
`check_findings` after every `evaluate`, and `Finding.__post_init__` and
`Evaluator.finding` do the rest.

**(a) Every finding carries at least one cause.** A finding with no cause is
an assertion with no evidence; `alarm_causes` is the ground truth that lets
anyone ask later *why* an episode fired, and nothing could answer it.

**(b) At most one finding per subject.** Two findings with one subject collide
on alarm identity. If eleven metrics stepped, that is one `step` finding with
eleven causes, not eleven findings — which is also why `trends` accumulates
into per-subject lists instead of returning as it goes.

**(c) Return an empty sequence for "nothing to report."** Not a finding at
level 0. A clean point writes nothing to the database at all; the driver logs
it and advances the watermark.

**(d) Skip measurements with a nonzero `code`,** unless watching the codes is
the algorithm's actual job, as above. `Measurement.ok` is the test.

### What `evaluate` is given

`point` is a [`Point`](base.py): `iid`, `sid`, the interval's run and time
range, and the measurements. Useful members:

| | |
|---|---|
| `point[vid]`, `point.get(vid)` | the `Measurement` — `valuex`, `sigma`, `code`, `.ok` |
| `point.vids` | the vids measured here, sorted |
| `point.limit(vid)` | the `Limit` **in force** at this point, or `None` |
| `point.name(vid)` | `"groupx/subgroup/namex"`, for logs and name-based config |
| `point.has_runs` | is this point positioned by run, or only by time? |
| `point.label` | short identification for a log line |
| `point.cause(vid, level=n)` | a `Cause` drawn from this point; `vid=None` is a global test |

`point.limit(vid)` is already resolved: `dqm.limits` is keyed per interval but
those are intervals of *validity*, so the band that applies is the most recent
committed one covering this point. The rule lives in [`../bands.py`](../bands.py),
shared with the display, so the band drawn on a timeline is the band the alarm
was judged against. An evaluator never has to do that lookup itself.

`history` holds up to `lookback` preceding points **in run/time order, oldest
first** — the physics timeline, not load order. Leave `lookback = 0` and it is
always empty. Three things about it:

* It may be shorter than asked for, early in a source's life. Cope with that;
  do not assume a full window. Both shipped evaluators that use it take a
  `min_history` par and say nothing below it.
* A config can override the class default (`lookback:` in the YAML), so the
  algorithm must still behave at other window sizes.
* Points where the producer flagged the measurement are *in* the history —
  filter them out yourself, as `TrendsEvaluator._series` does, or they widen
  the scale for no reason.

### Severity

`LEVELS` maps `message`/`warn`/`alarm`/`severe` to 0–3, from
`ALARM_LEVEL_NAMES` in `../schema.py` — one definition shared by the
evaluators, the display and the MCP, so write `LEVELS["warn"]` rather than a
bare `1`. Level belongs on the **cause**; an episode's level is the maximum
over its causes and escalates as more arrive. Where a band carries its own
`alarmcode`, prefer it over the configured level: it was set per metric and
deliberately. `LimitsEvaluator` does exactly that.

### Two lessons from `trends`, worth not relearning

A pure sigma test is unusable on this database. Many metrics are database
constants whose history is genuinely flat — for one CRV calibration constant,
100% of 1165 consecutive points are identical — so the robust scale collapses
to zero and *any* change is infinitely many sigma. Every threshold in `trends`
is therefore a pair: so many sigma **and** so large a fractional change. If a
new algorithm divides by a spread, it needs the same second gate.

And judge a slope against the scatter *about the line*, not the spread of the
window: a rising series has a large spread because it is rising, which hides
the very signal being looked for.

## 3. Register it

`REGISTRY` in [`config.py`](config.py) is a plain function returning a dict —
there will only ever be a handful of these, all in this package, and a
registry readable in one screen beat a plug-in mechanism:

```python
def _registry() -> Dict[str, type]:
    from dqm.evaluate.limits import LimitsEvaluator

    out = {LimitsEvaluator.algo: LimitsEvaluator}
    try:
        from dqm.evaluate.trends import TrendsEvaluator

        out[TrendsEvaluator.algo] = TrendsEvaluator
    except ImportError:
        pass
    try:                                        # <-- added
        from dqm.evaluate.codes import CodesEvaluator

        out[CodesEvaluator.algo] = CodesEvaluator
    except ImportError:
        pass
    return out
```

Import inside the function, not at module top: `config.py` is imported by
anything that reads a config, and an evaluator that needs an optional
dependency must not break the ones that do not. Without this step
`load_config` rejects the YAML with `unknown algo 'codes'`, which is the
intended failure — an unregistered algorithm cannot be run by accident.

## 4. Export it

In [`__init__.py`](__init__.py), beside the other two:

```python
from dqm.evaluate.codes import CodesEvaluator  # noqa: F401
```

and add `"CodesEvaluator"` to `__all__`. Nothing in the driver path needs
this; it is for `from dqm.evaluate import CodesEvaluator` in a notebook or a
test. Add a line for the new module to the inventory at the end of the
`__init__.py` docstring too.

## 5. The config and its working directory

One YAML, one evaluator, one working directory, one cron entry. Nothing is
shared, and two drivers cannot tread on each other — `load_state` refuses
outright to run if it finds a state file belonging to a different
`(algo, config, sid)`, because carrying on would record one evaluator's
progress against another's.

```yaml
# /exp/mu2e/app/users/rlc/dqm-drivers/crv-codes/crv-codes.yaml
algo:    codes
config:  crv-codes-0
sid:     115

commit:     false          # false = report only, nothing written
max_points: 200            # per invocation; startup walks forward
# lookback: 20             # only if the algorithm uses history

min_run:  125000           # where to start on the physics timeline
# min_time: 2026-10-01     # a date or a full timestamp; either, both, neither

workdir:   /exp/mu2e/app/users/rlc/dqm-drivers/crv-codes
log:       driver.log      # relative to workdir
log_level: INFO

pars:                      # whatever `configure()` reads
  level:    alarm
  fraction: 0.1
```

Set a floor. Without `min_run` or `min_time`, switching an evaluator on means
examining the entire history of the source first. The floor is about the
point's position on the timeline, not when it was committed, so a reprocessed
old file is skipped too — which is the intent.

`commit` defaults to false, so a driver run before anyone has thought about
the switch writes nothing.

## 6. Dry runs

```bash
dqm-driver crv-codes.yaml --check      # loads config + evaluator, then exits
dqm-driver crv-codes.yaml              # report mode, per the config
tail -f /exp/mu2e/app/users/rlc/dqm-drivers/crv-codes/driver.log
```

`--check` catches everything `configure()` validates and nothing about the
data. The report-mode run then logs `WOULD COMMIT subject=... level=... causes=N`
for each finding, with the full cause list at `log_level: DEBUG`.

Two things to know before iterating:

**The watermark advances in report mode too.** A dry run is still a run, and
re-examining the same points every invocation would never let the driver reach
the present. To re-examine, delete or restore `driver-state.json` in the
workdir — and `cp` it aside before switching to commit mode, since the
report-mode walk has already consumed the backlog.

**Nothing reaches the database without `dqmTool` on `PATH`** and a Kerberos
credential. The whole read-and-judge path is a plain pip install; only the
commit needs a Muse build. Report mode therefore works anywhere.

Testing without a database at all is the point of keeping the evaluator a pure
function: build a `Point` by hand and call `evaluate` directly.

```python
from dqm.evaluate.base import Measurement, Point
from dqm.evaluate.codes import CodesEvaluator

point = Point(iid=200, sid=115, start_run=125001,
              measurements={11: Measurement(11, 42.0, 0.5, 2)},
              names={11: "crv/digi/meanPulseHeight"})
ev = CodesEvaluator(config="test", sid=115, pars={"level": "alarm"})
assert [f.subject for f in ev.evaluate(point)] == ["missing"]
```

## 7. Commit, then cron

Flip `commit: true` (or pass `--commit` once by hand), re-run, and check
`dqm.alarms` — or the display's alarms tab — for the episode. Then:

```cron
17 * * * * cd /exp/mu2e/app/users/rlc/dqm-drivers/crv-codes && dqm-driver crv-codes.yaml
```

Exit status is 0 if the invocation did its work and 1 if anything failed,
which is what cron needs to decide whether to shout. Everything else is in the
log. If the batch filled, the log says `more to do, batch was full` and the
next invocation picks up where this one stopped — startup and keepup are the
same loop seeing different amounts of work, not two modes.

One episode behaviour to expect once it is live: a finding extends the
existing episode for `(algo, config, subject, sid)` rather than opening a new
one, for `ALARM_GRACE_HOURS` (24) after the last extension, measured on wall
clock. So the same subject firing on three consecutive points is one episode
with three causes. A finding arriving after the window opens a fresh episode,
as does one arriving after someone retired the old one.

## Pitfalls

* **A point past the watermark is never examined again.** If `evaluate` raises,
  the driver logs it, stops the batch and leaves the watermark short of the
  failure — deliberately, so nothing is lost. But a bug that returns a wrong
  answer quietly is permanent for those points.
* **Do not put the workdir anywhere with a comma in the path.** `dqmTool`
  decides whether `--cause` is an inline cause or a filename by looking for a
  comma, and a multi-cause finding has to go through a file. The driver checks
  and refuses rather than committing a malformed cause.
* **One workdir per driver**, per step 5.
* **`sigma = 0.0` means "not applicable"**, not zero error. Anything dividing
  by it needs a floor — `trends` takes the max of the robust spread, the quoted
  sigma and a configured `sigma_floor`.
* **Do not add a "stuck" check.** A metric frozen at one value is a real DQM
  failure, but in this database being frozen is the *normal* state for many
  metrics, so the check fires on nearly everything and means nothing.
  Recognising a genuinely stuck metric needs a comparison a single-metric
  evaluator cannot make — whether its siblings moved.
* **`note` is algo-internal JSON and is not queryable.** Useful detail for a
  human reading one episode; never the place to put something that has to be
  searched. That is what `subject`, `level` and the causes are for.
