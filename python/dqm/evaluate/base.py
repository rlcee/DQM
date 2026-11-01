"""The driver-evaluator contract: what a point is, what a finding is, and
the base class every evaluator implements.

An evaluator judges one timeline point at a time. The driver owns the loop,
the ordering and the bookkeeping, because those are the same for every
algorithm; the evaluator is left as a pure function of its inputs, testable
against a fixture with no database anywhere near it.

    class MyCheck(Evaluator):
        algo = "mycheck"
        subjects = ("too_high",)

        def evaluate(self, point, history):
            if point["someMetric"].valuex > self.pars["ceiling"]:
                return [self.finding("too_high", point.cause(vid, level=2))]
            return []

Why one point rather than a window.  An episode is identified by
(algo, config, subject, sid) -- no iid -- so a finding reported separately for
each of three intervals extends one episode and accumulates all three causes.
The database state is identical to reporting them together in a window, which
means the window buys nothing the schema can see, while costing every
evaluator its own windowing code and the driver its per-point progress.

Algorithms that genuinely need more than one point -- a drift, an outlier
against a recent distribution -- declare a `lookback` and receive those points
as `history`. The loop still belongs to the driver.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from dqm.bands import has_run
from dqm.errors import DqmConfigError, DqmError
from dqm.schema import ALARM_LEVEL_NAMES

LOGGER = logging.getLogger("dqm.evaluate")

__all__ = [
    "has_run",
    "Measurement",
    "Limit",
    "Point",
    "Cause",
    "Finding",
    "Evaluator",
    "DqmEvaluateError",
    "LEVELS",
    "LEVEL_NAMES",
]


class DqmEvaluateError(DqmError):
    """An evaluator or its configuration is wrong."""


# 0,1,2,3 = message, warn, alarm, severe.  From dqm.schema, which is where the
# shared contract lives: the display, the MCP and the evaluators all have to
# agree on what a level means, and a second copy here would be the obvious
# place for them to drift apart.  Named so an algorithm can say LEVELS["warn"]
# rather than a bare 1 that the reader has to look up.
LEVEL_NAMES: Tuple[str, ...] = ALARM_LEVEL_NAMES
LEVELS: Dict[str, int] = {name: i for i, name in enumerate(LEVEL_NAMES)}


@dataclass(frozen=True)
class Measurement:
    """One metric's value in one interval -- a row of dqm.numbers."""

    vid: int
    valuex: float
    sigma: float
    code: int

    @property
    def ok(self) -> bool:
        """True if the producer flagged this measurement as usable.

        code is a fixed enum: 0 OK, 1 lowStats, 2 missing, 3 error.  An
        algorithm should normally skip anything non-zero rather than alarm on
        it -- a missing measurement is the extractor's problem, not a physics
        finding.
        """
        return self.code == 0


@dataclass(frozen=True)
class Limit:
    """The acceptable band for one metric in one interval, if one is set."""

    vid: int
    llimit: Optional[float]
    ulimit: Optional[float]
    sigma: Optional[float]
    alarmcode: int


@dataclass(frozen=True)
class Point:
    """One timeline point: an interval, and everything measured in it.

    This is the unit of evaluation.  All of an interval's measurements are
    committed together, so a point is complete the first time the driver sees
    it -- there is no partial point to worry about.
    """

    iid: int
    sid: int
    start_run: Optional[int] = None
    start_subrun: Optional[int] = None
    end_run: Optional[int] = None
    end_subrun: Optional[int] = None
    start_time: Optional[dt.datetime] = None
    end_time: Optional[dt.datetime] = None
    measurements: Mapping[int, Measurement] = field(default_factory=dict)
    limits: Mapping[int, Limit] = field(default_factory=dict)
    # Metric names, for logging and for an algorithm configured by name rather
    # than by vid.  Not every caller fills this in.
    names: Mapping[int, str] = field(default_factory=dict)

    def __contains__(self, vid: int) -> bool:
        return vid in self.measurements

    def __getitem__(self, vid: int) -> Measurement:
        return self.measurements[vid]

    def get(self, vid: int) -> Optional[Measurement]:
        return self.measurements.get(vid)

    def limit(self, vid: int) -> Optional[Limit]:
        return self.limits.get(vid)

    def name(self, vid: int) -> str:
        return self.names.get(vid, f"vid={vid}")

    @property
    def vids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.measurements))

    @property
    def has_runs(self) -> bool:
        """True if this point is positioned by run rather than only by time."""
        return has_run(self.start_run)

    @property
    def label(self) -> str:
        """A short identification for a log line.

        A run number when there is a real one, otherwise the date.  A
        day-aggregation source stores run 0 for every interval, so preferring
        the run unconditionally labelled every nightly point "run 0:0", which
        identifies nothing.
        """
        if self.has_runs:
            return f"iid={self.iid} run {self.start_run}:{self.start_subrun or 0}"
        if self.start_time is not None:
            return f"iid={self.iid} {self.start_time:%Y-%m-%d}"
        return f"iid={self.iid}"

    def cause(self, vid: Optional[int] = None, level: int = 2) -> "Cause":
        """Evidence drawn from this point.

        vid=None is a global test -- one about the interval as a whole rather
        than about any single metric.
        """
        return Cause(iid=self.iid, vid=vid, level=level)


@dataclass(frozen=True)
class Cause:
    """One piece of evidence: what was seen, where, and how bad."""

    iid: int
    vid: Optional[int]
    level: int

    def __post_init__(self) -> None:
        if self.level not in range(len(LEVEL_NAMES)):
            raise DqmEvaluateError(
                f"level must be 0-{len(LEVEL_NAMES) - 1}, got {self.level}"
            )

    def as_csv(self) -> str:
        """The form dqmTool --cause wants: iid,vid,level (vid may be empty)."""
        return f"{self.iid},{'' if self.vid is None else self.vid},{self.level}"


@dataclass(frozen=True)
class Finding:
    """One thing an evaluator concluded, with the evidence for it.

    algo, config and sid are not here: they belong to the evaluator instance,
    and repeating them per finding would let them disagree.
    """

    subject: str
    causes: Tuple[Cause, ...]
    note: Optional[str] = None

    def __post_init__(self) -> None:
        # Spec rule (a): a finding without a cause is an assertion with no
        # evidence, and nothing downstream could ever explain why it fired.
        if not self.causes:
            raise DqmEvaluateError(f"finding {self.subject!r} has no causes")

    @property
    def level(self) -> int:
        """The episode's level is the maximum over its causes."""
        return max(c.level for c in self.causes)


class Evaluator(ABC):
    """Base class for every evaluator.

    Subclasses set `algo` and `subjects`, optionally `lookback`, and implement
    `evaluate`.  Everything else -- ordering, progress, retries, committing --
    is the driver's.

    An instance is fixed by (algo, config, sid) at construction, per the spec:
    those three plus a subject are what identify an alarm episode, so they
    cannot vary per call.
    """

    #: This algorithm's name, as stored in dqm.alarms.algo.
    algo: str = ""

    #: The subjects this algorithm can report.  A controlled vocabulary, so
    #: that queries do not silently miss rows when someone writes
    #: "gain shift" where the code wrote "gain_shifted".
    subjects: Tuple[str, ...] = ()

    #: How many preceding points this algorithm needs in order to judge one.
    #: Zero means it looks only at the point in front of it, and `history`
    #: will be empty.
    lookback: int = 0

    def __init__(self, config: str, sid: int, pars: Optional[Mapping[str, Any]] = None):
        if not self.algo:
            raise DqmConfigError(
                f"{type(self).__name__} does not set a class-level `algo`"
            )
        if not self.subjects:
            raise DqmConfigError(
                f"{type(self).__name__} does not declare its `subjects`"
            )
        if not config:
            raise DqmConfigError("an evaluator needs a config label")
        self.config = str(config)
        self.sid = int(sid)
        self.pars: Dict[str, Any] = dict(pars or {})
        self.configure()

    def configure(self) -> None:
        """Validate and normalise `pars`, once, at construction.

        Overriding this is how an evaluator rejects a bad config file at
        startup rather than on the first point it looks at -- a cron job that
        fails immediately is much easier to diagnose than one that runs for an
        hour and then throws.
        """

    @abstractmethod
    def evaluate(self, point: Point, history: Sequence[Point] = ()) -> Sequence[Finding]:
        """Judge one point, optionally in the light of the ones before it.

        `history` holds up to `lookback` preceding points in run/time order,
        oldest first -- the physics timeline, not the order they were loaded.
        It may be shorter than asked for, at the start of a source's life, and
        an algorithm must cope with that rather than assume a full window.

        Return an empty sequence when there is nothing to report.  At most one
        finding per subject, per spec rule (b): two findings with the same
        subject would collide on alarm identity.
        """
        raise NotImplementedError

    # -- helpers for subclasses -------------------------------------------

    def finding(self, subject: str, *causes: Cause, note: Optional[str] = None) -> Finding:
        """Build a finding, checking the subject against the vocabulary."""
        if subject not in self.subjects:
            raise DqmEvaluateError(
                f"{self.algo}: {subject!r} is not one of this algorithm's "
                f"subjects {self.subjects}"
            )
        flat: list = []
        for c in causes:
            if isinstance(c, Cause):
                flat.append(c)
            else:
                flat.extend(c)  # a sequence of causes, for convenience
        return Finding(subject=subject, causes=tuple(flat), note=note)

    def check_findings(self, findings: Iterable[Finding]) -> Tuple[Finding, ...]:
        """Enforce rule (b) -- at most one finding per subject."""
        out = tuple(findings)
        seen = [f.subject for f in out]
        dupes = {s for s in seen if seen.count(s) > 1}
        if dupes:
            raise DqmEvaluateError(
                f"{self.algo}/{self.config}: more than one finding for "
                f"subject(s) {sorted(dupes)}; they would collide on alarm "
                f"identity -- use distinct subjects"
            )
        return out

    def __str__(self) -> str:
        return f"{self.algo}/{self.config} sid={self.sid}"
