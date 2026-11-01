"""Which limit band governs a point.

dqm.limits is keyed per interval, but a limit's interval is one of validity,
not one of measurement: a band set at run 2112 with no end run governs every
later run until a newer band supersedes it. No interval in this database
carries both a limit and a measurement, so matching a measurement's iid
against a limit's iid finds nothing ever.

The rule: among the bands whose range covers a point, the one committed most
recently wins -- highest lid, lid being a sequence and therefore commit order.
Restating a band over a new range is how a limit is changed, and the newest
statement is the one that counts.

This lives here rather than inside the evaluator because the display has to
draw the band the evaluator judged against. A second copy of the rule would
drift, and a timeline drawing a band that no alarm ever used is worse than one
drawing no band at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = ["has_run", "as_int", "as_time", "covers", "newest_first", "in_force"]


def has_run(run: Optional[int]) -> bool:
    """True if this run number carries information.

    Run 0 never identifies real Mu2e data or simulation, so a stored 0 means
    "no run here" exactly as NULL does. A day-aggregation source stores 0 for
    every one of its intervals and keeps its real position in the timestamps,
    so treating 0 as a run makes every such interval compare equal.
    """
    return run is not None and run != 0


def as_int(value) -> Optional[int]:
    """None for a missing value, int otherwise.

    An integer column that is NULL anywhere arrives from pandas as a float,
    and 125205.0 in a log line or an SQL literal is wrong.
    """
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_time(value):
    """A usable timestamp, or None for anything that is not one.

    Both infinities read as None: '-infinity' is how an interval with no time
    range is stored, and an open-ended band's 'infinity' end means the same
    thing a missing end does. NaT arrives when a column could not be parsed.
    """
    if value is None:
        return None
    try:
        if value != value:  # NaN / NaT
            return None
    except TypeError:
        pass
    text = str(value)
    if text.startswith("-inf") or text.startswith("inf"):
        return None
    return value


def covers(band, run=None, subrun=None, time=None) -> bool:
    """True if this band's interval of validity contains the position given.

    By run when both the band and the position have real runs, otherwise by
    time. An end that is absent, infinite or unparseable leaves the band open
    to the right, which is what makes it an interval of validity rather than
    an annotation on one point.

    A band with no usable range cannot be shown to apply, so it does not:
    guessing would judge a point against the wrong band.
    """
    b_start_run = as_int(getattr(band, "start_run", None))
    if has_run(b_start_run) and has_run(as_int(run)):
        b_end_run = as_int(getattr(band, "end_run", None))
        lo = (b_start_run, as_int(getattr(band, "start_subrun", None)) or 0)
        here = (as_int(run), as_int(subrun) or 0)
        if here < lo:
            return False
        if not has_run(b_end_run):
            return True
        hi = (b_end_run, as_int(getattr(band, "end_subrun", None)) or 0)
        return here <= hi

    b_start = as_time(getattr(band, "start_time", None))
    here_time = as_time(time)
    if b_start is not None and here_time is not None:
        if here_time < b_start:
            return False
        b_end = as_time(getattr(band, "end_time", None))
        return b_end is None or here_time <= b_end

    return False


def newest_first(limits) -> List[Any]:
    """Band rows as a list, newest commit first.

    Sorted once by the caller that holds the frame rather than per point: a
    source may have hundreds of bands and thousands of points. Accepts a
    DataFrame or an already-ordered sequence.
    """
    if limits is None:
        return []
    if hasattr(limits, "empty"):
        if limits.empty:
            return []
        ordered = (limits.sort_values("lid", ascending=False)
                   if "lid" in limits.columns else limits)
        return list(ordered.itertuples())
    return list(limits)


def in_force(bands: Sequence[Any], run=None, subrun=None,
             time=None) -> Dict[int, Any]:
    """The band row governing each vid at the position given, by vid.

    `bands` must be newest-commit-first, as newest_first returns: the first
    covering row for a vid is the one in force and later ones are superseded.
    """
    out: Dict[int, Any] = {}
    for row in bands:
        vid = as_int(getattr(row, "vid", None))
        if vid is None or vid in out:
            continue
        if not covers(row, run=run, subrun=subrun, time=time):
            continue
        out[vid] = row
    return out
