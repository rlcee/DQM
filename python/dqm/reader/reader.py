"""Read the Mu2e DQM database through Query Engine.

This is the read path every DQM consumer shares: the dqm MCP server, the
display applications, the evaluators, and bulk retrospective or machine
learning analysis. It is deliberately capable -- it will pull the whole table
if asked -- and callers that must stay small, like the MCP, impose their own
limits rather than being constrained here.

Sizing drove the design. dqm.numbers holds ~475k rows and is expected to reach
millions. As per-row dicts that is roughly a kilobyte per row, so a million
rows would approach a gigabyte and tens of seconds to build; as a DataFrame
with numeric dtypes it is about 100 MB and a second or two, because read_csv
goes straight from the response stream into typed columns. So every query here
returns a DataFrame, and callers needing JSON convert at their own boundary
with dqm.frames.to_records.

Three properties matter for correctness at that scale:

Limits are explicit. QE returns the entire table when given no `l` parameter,
so every entry point defaults to DEFAULT_LIMIT. Pass limit=None to ask for
everything deliberately.

Truncation is labeled. A capped result that looks merely short is how "this
source has no data" gets reported about a source that has plenty. Frames carry
the dqm.frames truncation flag, set by fetching one row past the limit and
seeing whether it came back; count_metrics gives the exact total when a caller
needs to report one.

Filters are pushed to the database. A multi-id filter becomes one query per id,
because QE has no IN operator. The alternative -- scanning the most recent N
rows and filtering afterwards -- silently misses a low-rate source sharing a
table with high-rate ones, no matter how large the scan window, because the
window needed depends on every other source's insertion rate rather than on
this query's own selectivity.

Typical use:

    from dqm.reader import DqmReader

    r = DqmReader()
    src = r.sources(process="valNightly", stream="reco", aggregation="day")
    val = r.values(groupx="cal", namex="meanE0")
    df = r.numbers(sid=int(src.sid.iloc[0]), vid=int(val.vid.iloc[0]))

`df` is a timeline of one variable for one source, ready to plot. For something
too large to hold at once, iterate:

    for chunk in r.iter_metrics(sid=5, limit=None, chunksize=100_000):
        ...
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd

from dqm import schema
from dqm.errors import DqmConfigError, TooManyQueriesError
from dqm.bands import in_force, newest_first
from dqm.frames import empty_frame, is_truncated, preserve_attrs, tag
from dqm.schema import (
    ALARM_VALUES,
    CODE_VALUES,
    INTERVALS,
    METRIC_TABLES,
    SOURCES,
    VALUES,
    Table,
)
from dqm.timeutil import TimeLike, cutoff, qe_time, to_utc
from dqm.reader.transport import (
    DEFAULT_DBNAME,
    DEFAULT_QE_NOCACHE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    QEClient,
    where,
    where_eq,
)

LOGGER = logging.getLogger("dqm.reader")

__all__ = ["DqmReader", "DEFAULT_LIMIT", "DEFAULT_MAX_PUSHDOWN_QUERIES"]

# Applied by every entry point that does not get an explicit limit. Large
# enough that ordinary interactive work never notices it, small enough that an
# accidental unfiltered call cannot drag the whole table across the network.
DEFAULT_LIMIT = 100_000

# A filter matching several ids becomes one query per id. This bounds that
# fan-out; beyond it the caller is told to narrow rather than left waiting on
# hundreds of round trips.
DEFAULT_MAX_PUSHDOWN_QUERIES = 200

IdSpec = Union[None, int, Sequence[int]]


def _as_id_list(value: IdSpec) -> Optional[List[int]]:
    """Normalize an id argument accepting a scalar, a sequence, or None."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    ids = [int(v) for v in value]
    if not ids:
        raise DqmConfigError(
            "an explicit empty id list matches nothing; pass None for no filter"
        )
    return ids


class DqmReader:
    """A read connection to one DQM database.

    Construction takes no required arguments; the defaults are the production
    database over the Query Engine nocache endpoint. The overrides exist for
    testing against another database or a slower network.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_QE_NOCACHE_URL,
        dbname: str = DEFAULT_DBNAME,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_pushdown_queries: int = DEFAULT_MAX_PUSHDOWN_QUERIES,
        client: Optional[QEClient] = None,
    ) -> None:
        self.client = client or QEClient(
            base_url=base_url, dbname=dbname, timeout_seconds=timeout_seconds
        )
        self.max_pushdown_queries = max_pushdown_queries

    # Reading through rather than storing these keeps one copy of the truth,
    # so a caller inspecting the reader and one inspecting its client agree.
    @property
    def base_url(self) -> str:
        return self.client.base_url

    @property
    def dbname(self) -> str:
        return self.client.dbname

    @property
    def timeout_seconds(self) -> int:
        return self.client.timeout_seconds

    def __repr__(self) -> str:
        return f"DqmReader(dbname={self.dbname!r}, base_url={self.base_url!r})"

    # ---------------------------------------------------------------- low level

    def read_table(
        self,
        table: Union[str, Table],
        columns: Optional[Sequence[str]] = None,
        clauses: Optional[Sequence[str]] = None,
        order: Optional[str] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
        chunksize: Optional[int] = None,
    ):
        """Run one query against any DQM table and return a DataFrame.

        The escape hatch for anything the typed entry points below do not
        cover -- including dqm.alarms, dqm.alarm_causes and dqm.files, which
        have no dedicated method yet. `table` is a schema.Table or a bare name
        ("numbers", "alarms"); columns default to all of that table's.

        `clauses` are Query Engine where terms, built with transport.where to
        get the operator validated. `order` is a column name, optionally
        prefixed "-" for descending.

        limit=None fetches every matching row. With a limit, one extra row is
        requested so truncation is detected exactly rather than inferred from
        the count looking suspiciously round; the extra row is dropped and the
        frame is tagged.

        With chunksize, returns an iterator of frames and does no truncation
        bookkeeping -- a caller streaming does not need the total held or
        counted.
        """
        spec = schema.table(table) if isinstance(table, str) else table
        cols = list(columns) if columns else list(spec.columns)
        if limit is not None and limit <= 0:
            raise DqmConfigError("limit must be > 0, or None for no limit")

        if chunksize is not None:
            return self.client.query(
                spec.qualified, cols, clauses, order, limit, chunksize=chunksize
            )

        fetch_limit = None if limit is None else limit + 1
        df = self.client.query(spec.qualified, cols, clauses, order, fetch_limit)

        truncated = limit is not None and len(df) > limit
        if truncated:
            df = df.iloc[:limit]
        return tag(df, table=spec.qualified, limit=limit, truncated=truncated)

    def count(self, table: Union[str, Table], clauses: Optional[Sequence[str]] = None) -> int:
        """Count matching rows exactly.

        QE exposes no count aggregate, so this fetches the primary key column
        alone -- a few bytes per row -- and counts what returns. Cheap but not
        free: the whole current dqm.numbers key column is about three megabytes
        and under a second, scaling linearly. Use it to report "showing 100 of
        41,203", not inside a loop.
        """
        spec = schema.table(table) if isinstance(table, str) else table
        df = self.read_table(spec, [spec.pk], clauses=clauses, limit=None)
        return int(len(df))

    # ---------------------------------------------------------------- catalogs

    def sources(
        self,
        sid: IdSpec = None,
        process: Optional[str] = None,
        stream: Optional[str] = None,
        aggregation: Optional[str] = None,
        version: Optional[str] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """Monitored sources, keyed by (process, stream, aggregation, version).

        The sid is what scopes intervals and metrics to one source. A list of
        sids is matched client-side: QE has no IN operator and this catalog is
        small enough that filtering it locally is free.
        """
        sids = _as_id_list(sid)
        clauses: List[str] = []
        if sids is not None and len(sids) == 1:
            clauses.append(where_eq("sid", sids[0]))
        for column, value in (
            ("process", process),
            ("stream", stream),
            ("aggregation", aggregation),
            ("version", version),
        ):
            if value:
                clauses.append(where_eq(column, value))

        df = self.read_table(SOURCES, clauses=clauses or None, order="sid", limit=limit)
        if sids is not None and len(sids) > 1:
            df = self._isin(df, "sid", sids)
        return df

    def values(
        self,
        vid: IdSpec = None,
        groupx: Optional[str] = None,
        subgroup: Optional[str] = None,
        namex: Optional[str] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
    ) -> pd.DataFrame:
        """The catalog of trackable metric names, keyed by (groupx, subgroup, namex).

        Source-independent: a row here means the name exists, not that any
        particular source reports it.
        """
        vids = _as_id_list(vid)
        clauses: List[str] = []
        if vids is not None and len(vids) == 1:
            clauses.append(where_eq("vid", vids[0]))
        for column, value in (
            ("groupx", groupx),
            ("subgroup", subgroup),
            ("namex", namex),
        ):
            if value:
                clauses.append(where_eq(column, value))

        df = self.read_table(VALUES, clauses=clauses or None, order="vid", limit=limit)
        if vids is not None and len(vids) > 1:
            df = self._isin(df, "vid", vids)
        return df

    def intervals(
        self,
        sid: IdSpec = None,
        iid: IdSpec = None,
        run: Optional[int] = None,
        run_min: Optional[int] = None,
        run_max: Optional[int] = None,
        subrun: Optional[int] = None,
        start_time_after: Optional[TimeLike] = None,
        end_time_before: Optional[TimeLike] = None,
        recent_days: Optional[float] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
        newest_first: bool = True,
    ) -> pd.DataFrame:
        """Time/run buckets, each belonging to one source.

        `run` matches intervals whose run range contains it. `run_min` and
        `run_max` bound the interval's own start_run instead -- which is what
        "the data from runs X to Y" means in practice, and deliberately not
        the same as containment: some sources carry a catch-all interval
        spanning every run (start_run=44, end_run=999999) to hold limits, and
        a containment test would pull that into every run window.

        The time filters act on the interval's own start/end, and
        `recent_days` is shorthand for an interval ending within that many
        days of now.

        Not necessarily a small table: a 'file' aggregation source gets one
        interval per file.
        """
        sids = _as_id_list(sid)
        iids = _as_id_list(iid)

        clauses: List[str] = []
        if sids is not None and len(sids) == 1:
            clauses.append(where_eq("sid", sids[0]))
        if iids is not None and len(iids) == 1:
            clauses.append(where_eq("iid", iids[0]))
        elif iids is not None:
            clauses.extend(self._id_range("iid", iids))
        if run is not None:
            clauses.append(where("start_run", "le", run))
            clauses.append(where("end_run", "ge", run))
        if run_min is not None:
            clauses.append(where("start_run", "ge", run_min))
        if run_max is not None:
            clauses.append(where("start_run", "le", run_max))
        if subrun is not None:
            clauses.append(where("start_subrun", "le", subrun))
            clauses.append(where("end_subrun", "ge", subrun))
        if recent_days is not None:
            clauses.append(where("end_time", "ge", qe_time(cutoff(recent_days))))
        if start_time_after is not None:
            clauses.append(where("end_time", "ge", qe_time(start_time_after)))
        if end_time_before is not None:
            clauses.append(where("start_time", "le", qe_time(end_time_before)))

        df = self.read_table(
            INTERVALS,
            clauses=clauses or None,
            order="-iid" if newest_first else "iid",
            limit=limit,
        )
        if sids is not None and len(sids) > 1:
            df = self._isin(df, "sid", sids)
        if iids is not None and len(iids) > 1:
            df = self._isin(df, "iid", iids)
        return df

    def values_for_source(
        self,
        sid: int,
        table: str = "numbers",
        with_counts: bool = True,
    ) -> pd.DataFrame:
        """The metrics a source actually reports.

        dqm.values is deliberately source-independent -- it says a metric name
        exists, not that anything measures it -- so answering "what can I plot
        for this source?" means looking at what has actually been written.
        That is a question the catalog cannot answer and every display asks.

        Returns the catalog rows (vid, groupx, subgroup, namex) for the vids
        this source has measurements for, ordered by name. With `with_counts`
        it also carries `n`, the number of measurements behind each -- useful
        for telling a well-populated metric from one with a single stray row.

        Only the vid column crosses the network, so this costs about a fifth
        of a second on the busiest source today and scales with that source's
        row count, not with the whole table.
        """
        spec = self._metric_table(table)
        rows = self.read_table(
            spec, ["vid"], clauses=[where_eq("sid", int(sid))], limit=None
        )
        if rows.empty:
            return empty_frame(list(VALUES.columns) + (["n"] if with_counts else []),
                               schema.DTYPES)

        counts = rows["vid"].value_counts()
        vids = [int(v) for v in counts.index.tolist()]

        catalog = self.values(vid=vids, limit=None)
        if with_counts:
            catalog = catalog.copy()
            catalog["n"] = catalog["vid"].map(counts).astype("Int64")
        return catalog.sort_values(["groupx", "subgroup", "namex"]).reset_index(drop=True)

    def runs_for_source(self, sid: int) -> List[int]:
        """The run numbers this source has intervals for, newest first.

        What "the latest N runs" needs in order to mean anything: the run
        numbers that exist, rather than an arithmetic window around the
        highest one, since run numbers are not contiguous.

        Taken from each interval's start_run. A source with no run structure
        -- the nightly validation sources record every interval under run 0,
        because they are organised by date -- comes back as [0] or [], which
        is how a caller can tell that a run-based view would be meaningless
        for it.
        """
        iv = self.intervals(sid=int(sid), limit=None)
        if iv.empty:
            return []
        runs = {int(v) for v in iv["start_run"].dropna()}
        return sorted(runs, reverse=True)

    def runs_for_metric(self, sid: int, vid: int, table: str = "numbers") -> List[int]:
        """Runs for which this source reports this metric, newest first.

        Narrower than runs_for_source, and usually the one wanted. A source's
        newest runs are not necessarily the newest runs *for a given metric*:
        a metric can be retired, or added late, so the source may have run
        2112 while that metric stops at 1043. Asking for "the latest ten
        runs" of the source then selects a window the metric was never
        measured in, and the honest answer -- no data -- is not the one the
        question meant.

        Costs one query for the metric's interval ids plus the source's
        intervals, which the caller is likely to have cached already.
        """
        spec = self._metric_table(table)
        rows = self.read_table(
            spec,
            ["iid"],
            clauses=[where_eq("sid", int(sid)), where_eq("vid", int(vid))],
            limit=None,
        )
        if rows.empty:
            return []
        iv = self.intervals(sid=int(sid), limit=None)
        if iv.empty:
            return []
        run_by_iid = {
            int(i): int(r)
            for i, r in zip(iv["iid"], iv["start_run"])
            if not pd.isna(i) and not pd.isna(r)
        }
        runs = {run_by_iid[int(i)] for i in rows["iid"].dropna()
                if int(i) in run_by_iid}
        return sorted(runs, reverse=True)

    def sources_for_value(
        self,
        vid: int,
        table: str = "numbers",
        with_counts: bool = True,
    ) -> pd.DataFrame:
        """The sources that actually report a given metric -- the mirror of
        values_for_source, for "who measures this?"."""
        spec = self._metric_table(table)
        rows = self.read_table(
            spec, ["sid"], clauses=[where_eq("vid", int(vid))], limit=None
        )
        if rows.empty:
            return empty_frame(list(SOURCES.columns) + (["n"] if with_counts else []),
                               schema.DTYPES)

        counts = rows["sid"].value_counts()
        sids = [int(s) for s in counts.index.tolist()]

        catalog = self.sources(sid=sids, limit=None)
        if with_counts:
            catalog = catalog.copy()
            catalog["n"] = catalog["sid"].map(counts).astype("Int64")
        return catalog.sort_values(["process", "stream"]).reset_index(drop=True)

    @staticmethod
    def _isin(df: pd.DataFrame, column: str, ids: Sequence[int]) -> pd.DataFrame:
        """Exact client-side membership filter, keeping the frame's tags."""
        if df.empty:
            return df
        return preserve_attrs(df, df[df[column].isin(ids)].reset_index(drop=True))

    @staticmethod
    def _id_range(column: str, ids: Sequence[int]) -> List[str]:
        """A contiguous bound covering `ids`.

        The closest QE can express for a set of ids, since it has no IN
        operator. It narrows the fetch; exact membership is applied afterwards.
        """
        return [where(column, "ge", min(ids)), where(column, "le", max(ids))]

    # ------------------------------------------------------------- id resolution

    def resolve_sids(
        self,
        sid: IdSpec = None,
        process: Optional[str] = None,
        stream: Optional[str] = None,
        aggregation: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Turn source-name filters into a list of sids, or None for no filter.

        Returning None rather than "every sid" matters: it lets a caller with
        no source filter skip the per-sid fan-out and issue one query.
        """
        named = any((process, stream, aggregation, version))
        if not named:
            return _as_id_list(sid)
        df = self.sources(
            sid=sid,
            process=process,
            stream=stream,
            aggregation=aggregation,
            version=version,
            limit=None,
        )
        return [int(v) for v in df["sid"].dropna().tolist()]

    def resolve_vids(
        self,
        vid: IdSpec = None,
        groupx: Optional[str] = None,
        subgroup: Optional[str] = None,
        namex: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Turn value-name filters into a list of vids, or None for no filter."""
        named = any((groupx, subgroup, namex))
        if not named:
            return _as_id_list(vid)
        df = self.values(
            vid=vid, groupx=groupx, subgroup=subgroup, namex=namex, limit=None
        )
        return [int(v) for v in df["vid"].dropna().tolist()]

    # ------------------------------------------------------------- measurements

    def metrics(
        self,
        table: str = "numbers",
        sid: IdSpec = None,
        vid: IdSpec = None,
        process: Optional[str] = None,
        stream: Optional[str] = None,
        aggregation: Optional[str] = None,
        version: Optional[str] = None,
        groupx: Optional[str] = None,
        subgroup: Optional[str] = None,
        namex: Optional[str] = None,
        run: Optional[int] = None,
        run_min: Optional[int] = None,
        run_max: Optional[int] = None,
        subrun: Optional[int] = None,
        start_time_after: Optional[TimeLike] = None,
        end_time_before: Optional[TimeLike] = None,
        recent_days: Optional[float] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
        newest_first: bool = True,
        with_intervals: bool = True,
        decorate: bool = False,
    ) -> pd.DataFrame:
        """Measurements from dqm.numbers (default) or dqm.limits.

        The two tables share the (sid, vid, iid) key shape -- limits holds
        llimit/ulimit/alarmcode where numbers holds valuex/code -- so the same
        filters apply to either. A row in one does not imply a row in the
        other; they are populated independently with no foreign key between
        them.

        The meaningful query is one sid plus one vid: a timeline of one
        variable, ready to trend or plot. A sid alone returns every variable
        that source tracks interleaved, on unrelated scales.

        Source and value filters are pushed into the query as the cross product
        of the resolved ids. Time and run filters act on dqm.intervals, whose
        matching iids bound the metric query and are then applied exactly.

        with_intervals joins each row's run range and timestamps, which is what
        makes the result plottable. decorate additionally joins source and
        value names -- off by default because they are highly repetitive
        strings that, across a large result, cost more than the measurements
        themselves.
        """
        spec = self._metric_table(table)
        if limit is not None and limit <= 0:
            raise DqmConfigError("limit must be > 0, or None for no limit")

        sids, vids, iids, interval_df, impossible = self._resolve_filters(
            sid=sid,
            vid=vid,
            process=process,
            stream=stream,
            aggregation=aggregation,
            version=version,
            groupx=groupx,
            subgroup=subgroup,
            namex=namex,
            run=run,
            run_min=run_min,
            run_max=run_max,
            subrun=subrun,
            start_time_after=start_time_after,
            end_time_before=end_time_before,
            recent_days=recent_days,
            newest_first=newest_first,
            need_intervals=with_intervals or decorate,
        )
        if impossible:
            return self._empty_metrics(spec, with_intervals, decorate)

        df = self._fetch_metrics(spec, sids, vids, iids, limit, newest_first)

        if with_intervals or decorate:
            df = self.join_intervals(df, interval_df)
        if decorate:
            df = self.join_names(df, sids=sids, vids=vids)
        return df

    def numbers(self, **kwargs) -> pd.DataFrame:
        """Measurements from dqm.numbers. See metrics() for the arguments."""
        return self.metrics(table="numbers", **kwargs)

    def limits(self, **kwargs) -> pd.DataFrame:
        """Acceptable ranges from dqm.limits. See metrics() for the arguments."""
        return self.metrics(table="limits", **kwargs)

    def limits_in_force(self, sid: int, vid: IdSpec = None, run=None,
                        subrun=None, time=None, **kwargs) -> Dict[int, Any]:
        """The band governing each metric at one run or time, by vid.

        A limits row is not an annotation on one interval: its interval is one
        of validity, so the band that applies at a point is the most recently
        committed one whose range covers it. Asking dqm.limits for the rows
        near a point and reading any of them is wrong whenever a limit has
        been changed, which is why this is a query rather than a filter.

        Resolves one position. To follow a timeline, fetch the bands once with
        limits() and call dqm.bands.newest_first and in_force per point rather
        than paying for a query each.
        """
        # An ISO string is normalized here, at the boundary a caller supplies:
        # the per-point path passes datetimes already and should not pay for
        # this on every comparison.
        when = to_utc(time) if time is not None else None
        rows = self.limits(sid=sid, vid=vid, limit=None, **kwargs)
        return in_force(newest_first(rows), run=run, subrun=subrun, time=when)

    def count_metrics(
        self,
        table: str = "numbers",
        sid: IdSpec = None,
        vid: IdSpec = None,
        process: Optional[str] = None,
        stream: Optional[str] = None,
        aggregation: Optional[str] = None,
        version: Optional[str] = None,
        groupx: Optional[str] = None,
        subgroup: Optional[str] = None,
        namex: Optional[str] = None,
        run: Optional[int] = None,
        run_min: Optional[int] = None,
        run_max: Optional[int] = None,
        subrun: Optional[int] = None,
        start_time_after: Optional[TimeLike] = None,
        end_time_before: Optional[TimeLike] = None,
        recent_days: Optional[float] = None,
    ) -> int:
        """Count the rows metrics() would match, without fetching them.

        Takes the same filters. This is how a caller that has to cap its output
        says "showing 100 of 41,203" rather than leaving the reader of a
        truncated result to guess how much was left behind.

        Only key columns cross the network, so this is far cheaper than the
        fetch -- but still a scan, not an index lookup, since QE offers no
        count aggregate. Call it when a result came back truncated, not before
        every query.
        """
        spec = self._metric_table(table)
        sids, vids, iids, _, impossible = self._resolve_filters(
            sid=sid,
            vid=vid,
            process=process,
            stream=stream,
            aggregation=aggregation,
            version=version,
            groupx=groupx,
            subgroup=subgroup,
            namex=namex,
            run=run,
            run_min=run_min,
            run_max=run_max,
            subrun=subrun,
            start_time_after=start_time_after,
            end_time_before=end_time_before,
            recent_days=recent_days,
        )
        if impossible:
            return 0

        columns = [spec.pk, "iid"] if iids is not None else [spec.pk]
        total = 0
        for clauses in self._metric_queries(sids, vids, iids):
            part = self.read_table(spec, columns, clauses=clauses or None, limit=None)
            if iids is not None and not part.empty:
                part = part[part["iid"].isin(iids)]
            total += len(part)
        return total

    def iter_metrics(
        self,
        table: str = "numbers",
        sid: IdSpec = None,
        vid: IdSpec = None,
        limit: Optional[int] = None,
        chunksize: int = 100_000,
        newest_first: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """Stream metric rows in chunks, for results too large to hold at once.

        Deliberately narrower than metrics(): explicit ids only, no joins, no
        interval filter, because those all require holding or revisiting the
        whole result. Use it to aggregate across millions of rows without
        materializing them:

            total = sum(len(c) for c in r.iter_metrics(sid=5, chunksize=100_000))
        """
        spec = self._metric_table(table)
        sids = _as_id_list(sid)
        vids = _as_id_list(vid)
        if sids is not None and len(sids) > 1:
            raise DqmConfigError("iter_metrics takes at most one sid; loop over them yourself")
        if vids is not None and len(vids) > 1:
            raise DqmConfigError("iter_metrics takes at most one vid; loop over them yourself")

        clauses: List[str] = []
        if sids:
            clauses.append(where_eq("sid", sids[0]))
        if vids:
            clauses.append(where_eq("vid", vids[0]))

        return self.read_table(
            spec,
            clauses=clauses or None,
            order=f"-{spec.pk}" if newest_first else spec.pk,
            limit=limit,
            chunksize=chunksize,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _metric_table(table: str) -> Table:
        spec = METRIC_TABLES.get(table)
        if spec is None:
            raise DqmConfigError(
                f"table must be one of {sorted(METRIC_TABLES)}, not {table!r}"
            )
        return spec

    def _resolve_filters(
        self,
        sid: IdSpec,
        vid: IdSpec,
        process: Optional[str],
        stream: Optional[str],
        aggregation: Optional[str],
        version: Optional[str],
        groupx: Optional[str],
        subgroup: Optional[str],
        namex: Optional[str],
        run: Optional[int],
        subrun: Optional[int],
        start_time_after: Optional[TimeLike],
        end_time_before: Optional[TimeLike],
        recent_days: Optional[float],
        run_min: Optional[int] = None,
        run_max: Optional[int] = None,
        newest_first: bool = True,
        need_intervals: bool = False,
    ):
        """Turn name and time filters into id sets, for metrics and count alike.

        Returns (sids, vids, iids, interval_df, impossible). `impossible` marks
        a filter that matched no id at all: no metric row can satisfy it, so
        the empty answer is authoritative and the metric table need not be
        touched. A None id set means "no filter on this axis", which is not the
        same as an empty one.
        """
        sids = self.resolve_sids(sid, process, stream, aggregation, version)
        vids = self.resolve_vids(vid, groupx, subgroup, namex)

        if (sids is not None and not sids) or (vids is not None and not vids):
            return sids, vids, None, None, True

        by_interval = any(
            v is not None
            for v in (run, run_min, run_max, subrun, start_time_after,
                      end_time_before, recent_days)
        )
        if not (by_interval or need_intervals):
            return sids, vids, None, None, False

        interval_df = self.intervals(
            sid=sids,
            run=run,
            run_min=run_min,
            run_max=run_max,
            subrun=subrun,
            start_time_after=start_time_after,
            end_time_before=end_time_before,
            recent_days=recent_days,
            limit=None,
            newest_first=newest_first,
        )
        iids = None
        if by_interval:
            iids = [int(v) for v in interval_df["iid"].dropna().tolist()]
            if not iids:
                return sids, vids, iids, interval_df, True
        return sids, vids, iids, interval_df, False

    def _metric_queries(
        self,
        sids: Optional[List[int]],
        vids: Optional[List[int]],
        iids: Optional[List[int]],
    ) -> List[List[str]]:
        """One where-clause set per (sid, vid) pair to be pushed down.

        Both axes are pushed rather than one being filtered locally. The
        cheaper-looking alternative is wrong: the row limit would apply to the
        rows fetched rather than the rows matching, so asking for the 10 newest
        rows of one variable would fetch the 10 newest rows of *any* variable
        for that source and discard nearly all of them.
        """
        base: List[str] = []
        if iids is not None:
            base.extend(self._id_range("iid", iids))

        sid_terms: List[Optional[int]] = [None] if sids is None else list(sids)
        vid_terms: List[Optional[int]] = [None] if vids is None else list(vids)
        count = len(sid_terms) * len(vid_terms)
        if count > self.max_pushdown_queries:
            raise TooManyQueriesError(
                f"filter resolved to {len(sid_terms)} sid(s) x {len(vid_terms)} "
                f"vid(s) = {count} queries, over the max_pushdown_queries limit "
                f"of {self.max_pushdown_queries}. Query Engine has no IN "
                f"operator, so each pair costs its own round trip. Narrow the "
                f"filter, or raise max_pushdown_queries if the wait is "
                f"acceptable."
            )

        queries: List[List[str]] = []
        for one_sid in sid_terms:
            for one_vid in vid_terms:
                clauses = list(base)
                if one_sid is not None:
                    clauses.append(where_eq("sid", one_sid))
                if one_vid is not None:
                    clauses.append(where_eq("vid", one_vid))
                queries.append(clauses)
        return queries

    def _fetch_metrics(
        self,
        spec: Table,
        sids: Optional[List[int]],
        vids: Optional[List[int]],
        iids: Optional[List[int]],
        limit: Optional[int],
        newest_first: bool,
    ) -> pd.DataFrame:
        """Fetch metric rows, ordering and cutting the combined result once."""
        order = f"-{spec.pk}" if newest_first else spec.pk
        queries = self._metric_queries(sids, vids, iids)

        # The iid range pushed into the query is a superset of the intervals
        # wanted, so the exact filter below can still drop rows. When the query
        # is selective -- some sid or vid pinned -- fetch it whole and cut
        # afterwards, so the limit applies to matching rows. With nothing
        # pinned that could mean the entire table, so there the limit stays on
        # the query and a short result is reported as truncated.
        exact_iid = iids is not None
        selective = sids is not None or vids is not None
        query_limit = None if (exact_iid and selective) else limit

        frames: List[pd.DataFrame] = []
        truncated = False
        for clauses in queries:
            part = self.read_table(
                spec, clauses=clauses or None, order=order, limit=query_limit
            )
            # A capped fetch means rows exist beyond what was looked at,
            # whether or not the exact filter keeps the ones that came back.
            truncated = truncated or is_truncated(part)
            if exact_iid and not part.empty:
                part = part[part["iid"].isin(iids)]
            if not part.empty:
                frames.append(part)

        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = empty_frame(spec.columns, schema.DTYPES)

        # Each query came back ordered, but concatenating several breaks that,
        # so restore a single global order over the combined result.
        if not df.empty and len(queries) > 1:
            df = df.sort_values(spec.pk, ascending=not newest_first).reset_index(drop=True)
        if limit is not None and len(df) > limit:
            df = df.iloc[:limit].reset_index(drop=True)
            truncated = True

        return tag(df, table=spec.qualified, limit=limit, truncated=truncated)

    @staticmethod
    def _empty_metrics(spec: Table, with_intervals: bool, decorate: bool) -> pd.DataFrame:
        """An empty metric frame shaped like a populated one would be.

        Tagged not-truncated, because reaching here means no row could have
        matched -- the empty answer is authoritative.
        """
        columns = list(spec.columns)
        if with_intervals or decorate:
            columns += [c for c in INTERVALS.columns if c not in columns]
        if decorate:
            columns += [
                c
                for c in SOURCES.columns + VALUES.columns
                if c not in columns
            ]
        df = empty_frame(columns, schema.DTYPES)
        return tag(df, table=spec.qualified, limit=None, truncated=False)

    # ------------------------------------------------------------------- joins

    def join_intervals(
        self, df: pd.DataFrame, interval_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Attach each row's run range and timestamps from dqm.intervals.

        What turns a metric frame into something plottable, since the
        measurement rows carry only an iid.
        """
        if df.empty:
            return df
        if interval_df is None:
            iids = [int(v) for v in df["iid"].dropna().unique().tolist()]
            interval_df = self.intervals(iid=iids, limit=None)

        # sid is dropped: the metric frame already has it, and the interval's
        # copy would collide.
        columns = [c for c in INTERVALS.columns if c != "sid"]
        merged = df.merge(interval_df[columns], on="iid", how="left")
        return preserve_attrs(df, merged)

    def join_names(
        self,
        df: pd.DataFrame,
        sids: Optional[Sequence[int]] = None,
        vids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """Attach the source and value names behind each row's sid and vid.

        The joined columns are categoricals. Across a large result these are a
        handful of distinct strings repeated many times, and categories keep
        the cost proportional to the number of sources and metrics rather than
        to the number of rows.
        """
        if df.empty:
            return df
        if sids is None:
            sids = [int(v) for v in df["sid"].dropna().unique().tolist()]
        if vids is None:
            vids = [int(v) for v in df["vid"].dropna().unique().tolist()]

        merged = df.merge(self.sources(sid=list(sids), limit=None), on="sid", how="left")
        merged = merged.merge(self.values(vid=list(vids), limit=None), on="vid", how="left")
        for column in ("process", "stream", "aggregation", "version",
                       "groupx", "subgroup", "namex"):
            if column in merged.columns:
                merged[column] = merged[column].astype("category")
        return preserve_attrs(df, merged)

    # -------------------------------------------------------------- reductions

    @staticmethod
    def summarize(
        df: pd.DataFrame,
        value_column: str = "valuex",
        time_column: str = "end_time",
    ) -> Dict[str, Any]:
        """Reduce a metric frame to its shape: count, spread, span, endpoints.

        What a context-limited caller should ask for instead of paging: a
        forty-thousand-point timeline answers "is this drifting?" as well from
        its summary as from its rows.
        """
        summary: Dict[str, Any] = {
            "rows": int(len(df)),
            "truncated": is_truncated(df),
        }
        if df.empty:
            return summary

        if value_column in df.columns:
            series = pd.to_numeric(df[value_column], errors="coerce").dropna()
            if not series.empty:
                summary["value"] = {
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                }

        if time_column in df.columns and df[time_column].notna().any():
            times = df[time_column].dropna()
            summary["time"] = {
                "start": times.min().isoformat(),
                "end": times.max().isoformat(),
            }
            if value_column in df.columns:
                ordered = df.sort_values(time_column)[value_column].dropna()
                if not ordered.empty:
                    summary["first_value"] = float(ordered.iloc[0])
                    summary["last_value"] = float(ordered.iloc[-1])

        for column, mapping in (("code", CODE_VALUES), ("alarmcode", ALARM_VALUES)):
            if column in df.columns:
                counts = df[column].value_counts(dropna=True)
                summary[f"{column}_counts"] = {
                    mapping.get(int(k), str(k)): int(v) for k, v in counts.items()
                }
        return summary

    @staticmethod
    def downsample(
        df: pd.DataFrame,
        n: int = 200,
        value_column: str = "valuex",
        time_column: str = "end_time",
    ) -> pd.DataFrame:
        """Reduce a timeline to about n points, keeping its shape.

        Each output row is one time bin carrying the mean plus the min and max
        within it, so an excursion a plain stride would step over survives the
        reduction. Returns the frame unchanged when it is already small enough
        or has no usable time column to bin on.
        """
        if n <= 0:
            raise DqmConfigError("n must be > 0")
        if df.empty or len(df) <= n:
            return df
        if time_column not in df.columns or df[time_column].isna().all():
            return df.iloc[:n]

        work = df.dropna(subset=[time_column]).sort_values(time_column)
        if work.empty:
            return df.iloc[:n]

        bins = pd.cut(work[time_column], bins=n)
        grouped = work.groupby(bins, observed=True)

        out = pd.DataFrame(
            {time_column: grouped[time_column].mean(), "rows": grouped.size()}
        )
        if value_column in work.columns:
            binned = pd.to_numeric(work[value_column], errors="coerce").groupby(
                bins, observed=True
            )
            out[value_column] = binned.mean()
            out[f"{value_column}_min"] = binned.min()
            out[f"{value_column}_max"] = binned.max()

        out = out.reset_index(drop=True)
        preserve_attrs(df, out)
        out.attrs["downsampled_from"] = int(len(df))
        return out
