"""Turn database rows into Points for an evaluator to judge.

Two queries worth of joining happens here rather than in the driver, so that
the driver reads as a loop over points and the evaluator never sees a
DataFrame.

The ordering distinction this module exists to maintain:

  * New work is found in *insert* order, by nid.  A DQM file for an old run
    can be processed today -- in this database start_run decreases as iid
    increases in several places -- so run order would silently skip the late
    arrival.  nid comes from a sequence, so a watermark on it cannot.

  * History is assembled in *run/time* order.  A trend is about the physics
    timeline: when run 44 turns up late, the points before it are runs 43 and
    earlier, not whatever happened to be loaded first.

Both orderings are correct for their own job, and conflating them would break
one or the other.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from dqm.bands import (as_int as _as_int, as_time as _time, has_run,
                       in_force, newest_first)
from dqm.evaluate.base import Limit, Measurement, Point

LOGGER = logging.getLogger("dqm.evaluate.points")

__all__ = ["load_points", "load_history", "new_points", "History"]


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _resolve_limits(bands, point: Point) -> Dict[int, Limit]:
    """The band in force for each metric at this point, as Limits.

    The rule itself is in dqm.bands, shared with the display so that the band
    drawn on a timeline is the band an alarm was judged against.  This is only
    the adapter from a row to a Limit.
    """
    found = in_force(bands, run=point.start_run, subrun=point.start_subrun,
                     time=point.start_time)
    return {
        vid: Limit(
            vid=vid,
            llimit=_as_float(getattr(row, "llimit", None)),
            ulimit=_as_float(getattr(row, "ulimit", None)),
            sigma=_as_float(getattr(row, "sigma", None)),
            alarmcode=_as_int(getattr(row, "alarmcode", None)) or 0,
        )
        for vid, row in found.items()
    }


def _build(numbers, limits, names: Mapping[int, str]) -> List[Point]:
    """Group number rows by interval into Points, attaching the bands in force."""
    by_iid: Dict[int, List] = {}
    for row in numbers.itertuples():
        by_iid.setdefault(int(row.iid), []).append(row)

    bands = newest_first(limits)

    points: List[Point] = []
    for iid, rows in by_iid.items():
        first = rows[0]
        measurements = {
            int(r.vid): Measurement(
                vid=int(r.vid),
                valuex=float(r.valuex),
                sigma=_as_float(r.sigma) or 0.0,
                code=_as_int(r.code) or 0,
            )
            for r in rows
        }
        bare = Point(
            iid=iid,
            sid=int(first.sid),
            start_run=_as_int(getattr(first, "start_run", None)),
            start_subrun=_as_int(getattr(first, "start_subrun", None)),
            end_run=_as_int(getattr(first, "end_run", None)),
            end_subrun=_as_int(getattr(first, "end_subrun", None)),
            start_time=_time(getattr(first, "start_time", None)),
            end_time=_time(getattr(first, "end_time", None)),
            measurements=measurements,
            names=names,
        )
        # Which band applies depends on where the point sits, so it can only
        # be resolved once the point's range is known.
        points.append(replace(bare, limits=_resolve_limits(bands, bare)))
    return points


def _metric_names(reader, sid: int) -> Dict[int, str]:
    """vid -> "groupx/subgroup/namex", for logging and name-based config."""
    try:
        df = reader.values_for_source(sid)
    except Exception:
        LOGGER.debug("could not read metric names for sid=%s", sid, exc_info=True)
        return {}
    return {
        int(r.vid): f"{r.groupx}/{r.subgroup}/{r.namex}" for r in df.itertuples()
    }


def _with_geometry(reader, sid: int, rows):
    """Ensure the interval's run and time range are on each row.

    DqmReader.numbers already joins dqm.intervals, so normally there is
    nothing to do.  Merging unconditionally was a bug: it produced
    start_run_x / start_run_y and the geometry was then silently lost, every
    point arriving with no run number.  Only merge when the columns really are
    missing, so this keeps working whichever way the reader behaves.
    """
    if "start_run" in rows.columns or "start_time" in rows.columns:
        return rows
    try:
        intervals = reader.intervals(sid=sid, limit=None)
    except Exception:
        LOGGER.warning("could not attach interval ranges for sid=%s", sid,
                       exc_info=True)
        return rows
    drop = [c for c in ("sid",) if c in intervals.columns]
    return rows.merge(intervals.drop(columns=drop), on="iid", how="left")


def _eligible(rows, min_run: Optional[int], min_time):
    """Drop rows for points before the configured floor.

    Switching an evaluator on should not mean examining five years of water
    under the bridge.  The floor is about the point's position on the physics
    timeline -- its run or its date -- not about when it was committed, so a
    reprocessed old file is skipped too, which is the intent.

    A point that cannot be tested against the floor does not qualify.  Asking
    to start at a given run or date is a statement about what to report, so a
    point whose position in that dimension is unknown is not something to
    report on.  (The database holds a couple of hundred very early rows whose
    interval belongs to another source and which therefore have no position at
    all; they predate the current writer, which cannot produce more, and a
    floor excludes them along with everything else from that era.)
    """
    if min_run is None and min_time is None:
        return rows, 0

    keep = None
    if min_run is not None and "start_run" in rows.columns:
        keep = rows["start_run"].notna() & (rows["start_run"] >= min_run)
    if min_time is not None and "start_time" in rows.columns:
        by_time = rows["start_time"].notna() & (rows["start_time"] >= min_time)
        keep = by_time if keep is None else (keep & by_time)

    if keep is None:
        LOGGER.warning("a floor is configured but the rows carry neither "
                       "start_run nor start_time -- nothing skipped")
        return rows, 0

    dropped = int((~keep).sum())
    if dropped == len(rows):
        LOGGER.warning("the floor (min_run=%s min_time=%s) excluded every one "
                       "of %d new row(s) -- check it names the dimension this "
                       "source actually uses", min_run, min_time, len(rows))
    return rows[keep], dropped


def new_points(
    reader,
    sid: int,
    after_nid: int = 0,
    max_points: Optional[int] = None,
    with_limits: bool = True,
    min_run: Optional[int] = None,
    min_time=None,
) -> Tuple[List[Point], int]:
    """Points for `sid` whose measurements arrived after `after_nid`.

    Returns (points, high_nid).  `high_nid` is the largest nid safely
    accounted for, so cutting the batch at `max_points` leaves the rest to be
    found next time rather than skipped.

    Points come back in nid order, which is the order the measurements were
    committed.  That is deliberately not run order: see the module docstring.

    `min_run` and `min_time` skip points from before a chosen starting place
    on the physics timeline.  Skipped points still advance the watermark --
    they were excluded deliberately, and leaving them below it would mean
    re-reading and re-skipping them on every invocation forever.
    """
    numbers = reader.numbers(sid=sid, limit=None)
    if numbers.empty:
        return [], after_nid

    fresh = numbers[numbers["nid"] > after_nid]
    if fresh.empty:
        return [], after_nid

    eligible, skipped = _eligible(fresh, min_run, min_time)
    if skipped:
        LOGGER.info("skipping %d measurement row(s) before the configured "
                    "floor (min_run=%s min_time=%s)", skipped, min_run, min_time)
    if eligible.empty:
        # Everything new is below the floor: consume it so the next run does
        # not look at it again.
        return [], int(fresh["nid"].max())

    # One interval's measurements are committed together, so grouping by iid
    # and ordering the groups by their first nid reproduces commit order.
    order = eligible.groupby("iid")["nid"].min().sort_values()
    iids = list(order.index)
    if max_points is not None and len(iids) > max_points:
        iids = iids[:max_points]

    chosen = eligible[eligible["iid"].isin(iids)]

    # The watermark is the last nid that is safely accounted for: one below
    # the lowest *eligible* nid still waiting.  Not max(chosen), which would
    # be wrong the moment an interval's measurements were committed out of iid
    # order -- a reprocessed old file can do that -- and would then step over
    # the rows in between without examining them.
    remaining = eligible[~eligible["iid"].isin(iids)]
    if remaining.empty:
        # Nothing eligible left, so the skipped tail can be consumed as well.
        high_nid = int(fresh["nid"].max())
    else:
        high_nid = min(int(chosen["nid"].max()), int(remaining["nid"].min()) - 1)

    chosen = _with_geometry(reader, sid, chosen)

    limits = None
    if with_limits:
        try:
            limits = reader.limits(sid=sid, limit=None)
        except Exception:
            LOGGER.debug("no limits available for sid=%s", sid, exc_info=True)

    points = _build(chosen, limits, _metric_names(reader, sid))
    rank = {iid: i for i, iid in enumerate(iids)}
    points.sort(key=lambda p: rank.get(p.iid, 0))
    LOGGER.debug("sid=%s: %d point(s), nid %d..%d", sid, len(points),
                 int(chosen["nid"].min()), high_nid)
    return points, high_nid


def load_history(
    reader,
    sid: int,
    before: Point,
    count: int,
    with_limits: bool = True,
) -> List[Point]:
    """Up to `count` points preceding `before` on the physics timeline.

    Oldest first, which is what a trend wants to iterate.  Ordered by run then
    subrun when the source has runs, otherwise by time -- never by iid, so a
    late arrival sees the history that actually precedes it rather than
    whatever was loaded before it.

    May return fewer than asked for, at the start of a source's life.
    """
    if count <= 0:
        return []

    numbers = reader.numbers(sid=sid, limit=None)
    if numbers.empty:
        return []
    merged = _with_geometry(reader, sid, numbers)
    if "start_run" not in merged.columns and "start_time" not in merged.columns:
        LOGGER.warning("cannot order history without interval ranges for sid=%s", sid)
        return []

    by_run = before.has_runs and "start_run" in merged.columns
    if by_run:
        keys = ["start_run", "start_subrun"]
        here = (before.start_run, before.start_subrun or 0)
    else:
        keys = ["start_time"]
        here = (before.start_time,)

    geom = merged.dropna(subset=keys[:1]).drop_duplicates("iid").set_index("iid")
    if geom.empty:
        return []

    def sort_key(iid):
        row = geom.loc[iid]
        if by_run:
            return (_as_int(row["start_run"]) or 0, _as_int(row["start_subrun"]) or 0)
        return (row["start_time"],)

    candidates = [i for i in geom.index if i != before.iid]
    try:
        earlier = sorted((i for i in candidates if sort_key(i) < here), key=sort_key)
    except TypeError:
        # Mixed or unorderable keys -- better no history than a wrong one.
        LOGGER.warning("cannot order history for sid=%s around %s", sid, before.label)
        return []

    wanted = earlier[-count:]
    if not wanted:
        return []

    chosen = merged[merged["iid"].isin(wanted)]
    limits = None
    if with_limits:
        try:
            limits = reader.limits(sid=sid, limit=None)
        except Exception:
            pass
    points = _build(chosen, limits, before.names)
    rank = {iid: i for i, iid in enumerate(wanted)}
    points.sort(key=lambda p: rank.get(p.iid, 0))
    return points


def load_points(reader, sid: int, iids: Sequence[int]) -> List[Point]:
    """Specific points by iid, for re-examining something by hand."""
    if not iids:
        return []
    numbers = reader.numbers(sid=sid, limit=None)
    chosen = numbers[numbers["iid"].isin(list(iids))]
    if chosen.empty:
        return []
    chosen = _with_geometry(reader, sid, chosen)
    try:
        limits = reader.limits(sid=sid, limit=None)
    except Exception:
        limits = None
    return _build(chosen, limits, _metric_names(reader, sid))


class History:
    """Pre-loaded measurements for one source, for repeated history lookups.

    load_history on its own re-reads every measurement for the source each
    time it is called.  That is fine once and ruinous in a loop: a driver with
    a lookback spent twenty seconds on ten points, nearly all of it fetching
    the same three thousand rows ten times over.  Startup over a few thousand
    points would have taken hours.

    So the driver builds one of these per invocation.  The measurements cannot
    change underneath it -- a run examines a fixed batch, and anything
    committed meanwhile is simply next time's work -- so one fetch is both
    faster and more consistent than many.
    """

    def __init__(self, reader, sid: int, with_limits: bool = True):
        self.sid = int(sid)
        self._names = _metric_names(reader, sid)
        numbers = reader.numbers(sid=sid, limit=None)
        self._rows = _with_geometry(reader, sid, numbers) if not numbers.empty else numbers
        self._bands = None
        if with_limits:
            try:
                self._bands = reader.limits(sid=sid, limit=None)
            except Exception:
                LOGGER.debug("no limits for sid=%s", sid, exc_info=True)
        self._order = self._build_order()

    def _build_order(self):
        """iids in physics order, with the key each was sorted on."""
        if self._rows is None or self._rows.empty:
            return []
        cols = self._rows.columns
        # By run only if this source is actually positioned by run.  Ordering
        # a day-aggregation source by its run-0 placeholders made every key
        # equal, so nothing was ever earlier than anything and history came
        # back empty for every point.
        #
        # The test is a majority, not "any": valNightly/reco has two stray
        # intervals carrying run 1000 among sixteen hundred at run 0, and
        # "any" let those two decide the ordering for the whole source.  The
        # split is in practice stark -- 0.1% against 100% -- so a majority
        # separates the two kinds without a judgement call.
        by_run = False
        if "start_run" in cols:
            runs = self._rows.drop_duplicates("iid")["start_run"]
            informative = int(runs.apply(lambda v: has_run(_as_int(v))).sum())
            by_run = len(runs) > 0 and informative > len(runs) / 2
        if by_run:
            geom = self._rows.dropna(subset=["start_run"]).drop_duplicates("iid")
            keyed = [
                (int(r.iid), (_as_int(r.start_run) or 0,
                              _as_int(getattr(r, "start_subrun", None)) or 0))
                for r in geom.itertuples()
            ]
        elif "start_time" in cols:
            geom = self._rows.dropna(subset=["start_time"]).drop_duplicates("iid")
            keyed = [(int(r.iid), (r.start_time,)) for r in geom.itertuples()]
        else:
            return []
        try:
            keyed.sort(key=lambda kv: kv[1])
        except TypeError:
            LOGGER.warning("cannot order history for sid=%s", self.sid)
            return []
        return keyed

    def before(self, point: Point, count: int) -> List[Point]:
        """Up to `count` points preceding `point` on the physics timeline.

        Oldest first.  Ordered by run, or by time for a source with no runs --
        never by iid, so a late arrival sees the history that actually
        precedes it.
        """
        if count <= 0 or not self._order:
            return []
        if point.has_runs:
            here = (point.start_run, point.start_subrun or 0)
        elif point.start_time is not None:
            here = (point.start_time,)
        else:
            return []

        wanted = []
        for iid, key in self._order:
            if iid == point.iid:
                continue
            try:
                if key < here:
                    wanted.append(iid)
                else:
                    break  # self._order is sorted, so nothing later qualifies
            except TypeError:
                return []
        wanted = wanted[-count:]
        if not wanted:
            return []

        chosen = self._rows[self._rows["iid"].isin(wanted)]
        points = _build(chosen, self._bands, self._names)
        rank = {iid: i for i, iid in enumerate(wanted)}
        points.sort(key=lambda p: rank.get(p.iid, 0))
        return points
