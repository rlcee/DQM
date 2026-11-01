"""Timelines tab: several metrics as stacked plots, against time or run.

The natural DQM query is a single (source, metric) pair across intervals, and
that is what dqm.reader is built around. Comparing two of them -- the same
metric from two processing versions, or two sectors of the same detector -- is
the next thing anyone wants, so this tab draws several at once.

Each timeline gets its own panel rather than sharing one set of axes. Metrics
differ by orders of magnitude -- a CPU time beside a mean energy -- and a
shared y scale flattens the smaller into a line along the axis. The panels do
share an x axis, so they stay aligned and zooming one zooms all; that is what
makes a stack a comparison rather than a pile of unrelated plots.

What is shared and what is not follows from what a comparison means. The x
axis and its window are global; the source and metric are per panel, because
varying them is the entire point.

Rows are built with pattern-matching ids, so one callback serves however many
there are rather than one callback per row. Every row exists from the start
and surplus ones are hidden, which keeps the callback graph static -- Dash
cannot add callbacks after the app is running.

A dashboard is a saved arrangement of this tab -- which metrics, against what
axis, over what window -- read from a JSON file at startup. Selecting one
fills in every control at once. See dqm.web.dashboards.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import ALL, Input, Output, State, dcc, html

from dqm.bands import in_force, newest_first
from dqm.plot import PlotSpec, to_plotly_grid
from dqm.plot.spec import PALETTE
from dqm.schema import CODE_VALUES
from dqm.web.common import (ID, REFRESH_MS, card, drawn_stamp, error_box,
                            has_run_structure, reader, row, runs_for_metric,
                            source_reports, values_for_source)
from dqm.web.dashboards import load_dashboards, resolve_series

LOGGER = logging.getLogger("dqm.web.timeline")

PREFIX = "timeline-"

# How many panels can be stacked at once. Bounded by the palette -- a seventh
# colour would repeat, so two panels would share one -- and by how much fits
# on a screen before scrolling defeats the point of comparing them.
MAX_SERIES = 6

# Where the tab opens: valNightly/reco, a daily source with a long history,
# showing its calorimeter cluster mean energy. Both are validated against what
# the source actually reports before use, so neither becomes a stale constant
# if the database changes underneath them.
DEFAULT_SID = 5
DEFAULT_VID = 15

DEFAULT_DAYS = "365"
DEFAULT_RUNS = "50"

# Upper bounds on the window field. Not database limits -- they exist so a
# stray keypress asks for a decade rather than a millennium, and so the field
# cannot request an unbounded scan by accident.
MAX_DAYS = 20000
MAX_RUNS = 100000

MODES = {
    "time": {"label": "Days", "default": DEFAULT_DAYS, "unit": "days",
             "axis": "interval end", "blank": "all time"},
    "run": {"label": "Runs", "default": DEFAULT_RUNS, "unit": "runs",
            "axis": "run", "blank": "all runs"},
}

# Read once at import, so a hand-edited file is picked up on restart rather
# than on every interaction. A missing or malformed file yields an empty list
# and the dropdown simply does not appear -- the tab works without it.
DASHBOARDS = load_dashboards()
DASHBOARDS_BY_NAME = {d.name: d for d in DASHBOARDS}


def sid_id(index: int) -> dict:
    return {"type": PREFIX + "sid", "index": index}


def vid_id(index: int) -> dict:
    return {"type": PREFIX + "vid", "index": index}


def _source_options() -> List[dict]:
    df = reader().sources(limit=None)
    return [
        {
            "label": f"{r.process}/{r.stream}/{r.aggregation} v{r.version}",
            "value": int(r.sid),
        }
        for r in df.sort_values(["process", "stream", "version"]).itertuples()
    ]


def _value_options(sid: Optional[int]) -> List[dict]:
    """Metric options, restricted to what that source actually reports.

    dqm.values is source-independent, so the unrestricted catalog offers
    metrics a given source has never measured -- picking one yields an empty
    plot that looks like missing data rather than a bad question.
    """
    if sid is None:
        return []
    return [
        {
            # The count tells a well-populated metric from one with a couple
            # of stray rows -- worth knowing before plotting it.
            "label": f"{v.groupx}/{v.subgroup}/{v.namex}  ({v.n})",
            "value": int(v.vid),
        }
        for v in values_for_source(int(sid)).itertuples()
    ]


def _parse_window(raw, mode: str) -> Tuple[Optional[float], str]:
    """Turn the window field into a number, or explain why it will not.

    An **empty field is not an error** -- it means no window at all, so every
    point is shown. That is the honest reading of "no limit given".
    """
    if raw is None or str(raw).strip() == "":
        return None, ""

    unit = MODES[mode]["unit"]
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None, f"{raw!r} is not a number of {unit}."
    if value <= 0:
        return None, f"{unit.capitalize()} must be greater than zero."
    ceiling = MAX_DAYS if mode == "time" else MAX_RUNS
    if value > ceiling:
        return None, f"{value:g} {unit} is beyond the data; the maximum is {ceiling:,}."
    return value, ""


def _band_steps(ordered, key: str, vid: int, bands) -> Tuple[List, List]:
    """The llimit and ulimit staircases across a metric's points.

    A limit is not one number for the whole panel: committing a new band is
    how a limit is changed, so the band in force steps partway along. Each
    point is resolved where it sits -- by the same rule the evaluator judged
    it by, so the drawn band is the one that did or did not alarm.

    An x is repeated at each change to give the riser, since `lines` joins its
    points directly and would otherwise slope between two bands as though the
    limit had drifted.
    """
    lo: List[Tuple[Any, float]] = []
    hi: List[Tuple[Any, float]] = []
    previous = None
    for point in ordered.itertuples():
        band = in_force(bands,
                        run=getattr(point, "start_run", None),
                        subrun=getattr(point, "start_subrun", None),
                        time=getattr(point, "start_time", None)).get(vid)
        if band is None:
            # No band covers this point; leave a hole rather than stretching
            # a neighbouring band over it.
            previous = None
            continue
        here = (float(band.llimit), float(band.ulimit))
        x = getattr(point, key)
        if previous is not None and here != previous:
            lo.append((x, previous[0]))
            hi.append((x, previous[1]))
        lo.append((x, here[0]))
        hi.append((x, here[1]))
        previous = here
    return lo, hi


def _swatch(index: int) -> Any:
    """The colour this row will be drawn in.

    Shown beside the controls so a row can be matched to its curve without
    reading the legend -- which is what keeps the rows compact enough to want
    several of them.
    """
    return html.Span(
        style={
            "display": "inline-block", "width": "0.85rem", "height": "0.85rem",
            "borderRadius": "2px", "background": PALETTE[index % len(PALETTE)],
            "flex": "0 0 auto",
        }
    )


def _series_row(index: int, sources: List[dict], visible: bool,
                sid: Optional[int] = None, vid: Optional[int] = None,
                options: Optional[List[dict]] = None) -> Any:
    """One (source, metric) selector.

    Deliberately terse: a swatch and two dropdowns on a single line, with the
    column headings printed once above rather than on every row. Six rows of
    labelled fields would push the plot off the screen.
    """
    return html.Div(
        [
            _swatch(index),
            html.Div(
                dcc.Dropdown(
                    id=sid_id(index), options=sources, value=sid,
                    placeholder="source", clearable=True,
                    style={"fontSize": "0.85rem"},
                ),
                style={"flex": "1 1 250px", "minWidth": "180px"},
            ),
            html.Div(
                dcc.Dropdown(
                    id=vid_id(index), options=options or [], value=vid,
                    placeholder="metric", clearable=True,
                    style={"fontSize": "0.85rem"},
                ),
                style={"flex": "1 1 290px", "minWidth": "200px"},
            ),
        ],
        id=ID(PREFIX, f"row-{index}"),
        style={
            "display": "flex" if visible else "none",
            "gap": "0.5rem", "alignItems": "center", "marginBottom": "0.3rem",
        },
    )


def layout() -> Any:
    sources = _source_options()
    try:
        first_options = _value_options(DEFAULT_SID)
    except Exception:  # a database hiccup should not stop the tab rendering
        LOGGER.exception("could not list metrics for the default source")
        first_options = []
    first_vid = DEFAULT_VID if any(o["value"] == DEFAULT_VID for o in first_options) \
        else (first_options[0]["value"] if first_options else None)

    return html.Div(
        [
            card(
                [
                    # Global controls: what the x axis is, how much of it to
                    # show, how many series, and how to draw them.
                    row(
                        [
                            *([
                                html.Div(
                                    [
                                        html.Label("Dashboard",
                                                   style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id=ID(PREFIX, "dashboard"),
                                            options=[{"label": d.name, "value": d.name}
                                                     for d in DASHBOARDS],
                                            value=None,
                                            placeholder="(none)",
                                            clearable=True,
                                            style={"fontSize": "0.85rem"},
                                        ),
                                    ],
                                    style={"flex": "1 1 220px", "minWidth": "180px"},
                                )
                            ] if DASHBOARDS else []),
                            html.Div(
                                [
                                    html.Label("X axis", style={"fontWeight": 600}),
                                    dcc.RadioItems(
                                        id=ID(PREFIX, "mode"),
                                        options=[{"label": " time", "value": "time"},
                                                 {"label": " run", "value": "run"}],
                                        value="time", inline=True,
                                        inputStyle={"marginRight": "0.25rem"},
                                        labelStyle={"marginRight": "0.6rem"},
                                    ),
                                ],
                                style={"flex": "0 0 135px"},
                            ),
                            html.Div(
                                [
                                    html.Label(MODES["time"]["label"],
                                               id=ID(PREFIX, "window-label"),
                                               style={"fontWeight": 600}),
                                    dcc.Input(
                                        id=ID(PREFIX, "window"),
                                        # Text, not type="number": the spinner
                                        # arrows cannot be removed reliably
                                        # across browsers, and with debounce a
                                        # spinner click makes Dash emit None,
                                        # blanking the field.
                                        type="text", inputMode="numeric",
                                        value=DEFAULT_DAYS, debounce=True,
                                        placeholder=DEFAULT_DAYS,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"flex": "0 0 90px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Timelines", style={"fontWeight": 600}),
                                    dcc.Dropdown(
                                        id=ID(PREFIX, "count"),
                                        options=[{"label": str(n), "value": n}
                                                 for n in range(1, MAX_SERIES + 1)],
                                        value=1, clearable=False,
                                    ),
                                ],
                                style={"flex": "0 0 95px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Options", style={"fontWeight": 600}),
                                    dcc.Checklist(
                                        id=ID(PREFIX, "opts"),
                                        options=[{"label": " limits", "value": "limits"},
                                                 {"label": " log y", "value": "logy"}],
                                        value=["limits"], inline=True,
                                    ),
                                ],
                                style={"flex": "0 0 165px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Refresh", style={"fontWeight": 600}),
                                    # Its own control rather than another entry
                                    # in "opts": a dashboard sets opts, and
                                    # choosing one should not quietly switch
                                    # auto-refresh off (or on).
                                    dcc.Checklist(
                                        id=ID(PREFIX, "auto"),
                                        options=[{"label": " auto (10 min)",
                                                  "value": "auto"}],
                                        value=[], inline=True,
                                    ),
                                ],
                                style={"flex": "0 0 150px"},
                            ),
                        ]
                    ),
                    html.Hr(style={"margin": "0.6rem 0", "border": "none",
                                   "borderTop": "1px solid #eee"}),
                    # Column headings once, rather than a label per row.
                    html.Div(
                        [
                            html.Span(style={"flex": "0 0 0.85rem"}),
                            html.Div("Source", style={"flex": "1 1 250px",
                                                      "minWidth": "180px",
                                                      "fontWeight": 600,
                                                      "fontSize": "0.85rem"}),
                            html.Div("Metric", style={"flex": "1 1 290px",
                                                      "minWidth": "200px",
                                                      "fontWeight": 600,
                                                      "fontSize": "0.85rem"}),
                        ],
                        style={"display": "flex", "gap": "0.5rem",
                               "marginBottom": "0.3rem"},
                    ),
                    *[
                        _series_row(
                            i, sources, visible=(i == 0),
                            sid=DEFAULT_SID if i == 0 else None,
                            vid=first_vid if i == 0 else None,
                            options=first_options if i == 0 else [],
                        )
                        for i in range(MAX_SERIES)
                    ],
                ]
            ),
            # Always present, so the callback below always has this input to
            # bind to even when no dashboards are configured.
            *([] if DASHBOARDS else
              [dcc.Store(id=ID(PREFIX, "dashboard"), data=None)]),
            html.Div(id=ID(PREFIX, "dashboard-note"),
                     style={"fontSize": "0.8rem", "color": "#8a6d00",
                            "margin": "0.25rem 0"}),
            # Starts disabled; the callback below enables it when the box is
            # ticked.  A disabled Interval does not fire at all, so an idle
            # page costs nothing.
            dcc.Interval(id=ID(PREFIX, "tick"), interval=REFRESH_MS,
                         disabled=True, n_intervals=0),
            html.Div(id=ID(PREFIX, "status"), style={"margin": "0.5rem 0"}),
            dcc.Loading(
                dcc.Graph(id=ID(PREFIX, "graph")),
                type="default",
                # Dim the old figure rather than removing it: the controls
                # keep their position and the page does not jump mid-query.
                overlay_style={"visibility": "visible", "opacity": 0.5},
                delay_show=300,
            ),
            html.Div(id=ID(PREFIX, "links"), style={"marginTop": "0.5rem"}),
        ]
    )



def _report_href(ext: str, qs: str) -> str:
    """A link to a static export, as the browser must ask for it.

    Dash's get_relative_path prepends requests_pathname_prefix -- the path the
    browser uses, which behind a proxy is not the path this process serves on.
    A bare "/report/..." was correct only when the app sat at a hostname root;
    under a proxy path it pointed outside the app and 404ed.
    """
    try:
        import dash

        return dash.get_relative_path(f"/report/timeline.{ext}") + qs
    except Exception:
        # get_relative_path needs an app context; fall back to the bare path,
        # which is right whenever there is no prefix.
        return f"/report/timeline.{ext}{qs}"


def register(app) -> None:
    @app.callback(
        Output({"type": PREFIX + "vid", "index": ALL}, "options"),
        Output({"type": PREFIX + "vid", "index": ALL}, "value"),
        Output({"type": PREFIX + "sid", "index": ALL}, "value"),
        Output(ID(PREFIX, "count"), "value"),
        Output(ID(PREFIX, "mode"), "value"),
        Output(ID(PREFIX, "window"), "value"),
        Output(ID(PREFIX, "opts"), "value"),
        Output(ID(PREFIX, "dashboard-note"), "children"),
        Input({"type": PREFIX + "sid", "index": ALL}, "value"),
        Input(ID(PREFIX, "dashboard"), "value"),
        State({"type": PREFIX + "vid", "index": ALL}, "value"),
    )
    def _fill_rows(sids, dashboard_name, vids):
        """Keep each row's metric list in step with its source, and apply a
        dashboard when one is chosen.

        These are one callback rather than two on purpose. Applying a
        dashboard sets both the source and the metric of every row; if a
        separate callback owned the metric list, setting the source would
        fire it in parallel and it would reset the metric to that source's
        first -- discarding exactly the choice the dashboard was expressing.
        One callback decides both, so there is no race to lose.

        It also means a source change recomputes every row's options rather
        than just its own. The lookup is cached, so that costs nothing.
        """
        triggered = dash.callback_context.triggered_id
        by_dashboard = triggered == ID(PREFIX, "dashboard")

        n = len(sids)
        keep_sids = [dash.no_update] * n
        note = ""

        if by_dashboard and dashboard_name:
            board = DASHBOARDS_BY_NAME.get(dashboard_name)
            if board is None:
                return ([dash.no_update] * n, [dash.no_update] * n, keep_sids,
                        dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update, f"unknown dashboard {dashboard_name!r}")

            r = reader()
            pairs, problems = [], []
            for spec in board.series:
                sid, vid, problem = resolve_series(spec, r)
                if problem:
                    problems.append(problem)
                    continue
                pairs.append((sid, vid))

            if not pairs:
                return ([dash.no_update] * n, [dash.no_update] * n, keep_sids,
                        dash.no_update, dash.no_update, dash.no_update,
                        dash.no_update,
                        "This dashboard has no usable series: " + "; ".join(problems))

            if len(pairs) > n:
                problems.append(
                    f"only the first {n} of {len(pairs)} series fit; "
                    f"MAX_SERIES is {n}")
                pairs = pairs[:n]

            # Rows beyond the dashboard's series are cleared rather than left
            # holding a previous selection, which would reappear if the count
            # were raised afterwards.
            sids = [p[0] for p in pairs] + [None] * (n - len(pairs))
            vids = [p[1] for p in pairs] + [None] * (n - len(pairs))
            keep_sids = list(sids)
            count = len(pairs)
            mode = board.mode
            window = board.window_text
            opts = list(board.options)
            note = "; ".join(problems)
        else:
            count = mode = window = opts = dash.no_update

        options, values = [], []
        for i, sid in enumerate(sids):
            if sid is None:
                options.append([])
                values.append(None)
                continue
            try:
                row_options = _value_options(int(sid))
            except Exception:
                LOGGER.exception("could not list metrics for sid=%s", sid)
                options.append([])
                values.append(None)
                continue
            options.append(row_options)
            available = {o["value"] for o in row_options}
            wanted = vids[i] if i < len(vids) else None
            # Keep the metric when the source reports it -- so flipping
            # between sources that share one does not lose your place, and a
            # dashboard's choice survives -- otherwise fall back to the first.
            if wanted in available:
                values.append(wanted)
            elif row_options:
                values.append(row_options[0]["value"])
            else:
                values.append(None)

        return options, values, keep_sids, count, mode, window, opts, note

    @app.callback(
        [Output(ID(PREFIX, f"row-{i}"), "style") for i in range(MAX_SERIES)],
        Input(ID(PREFIX, "count"), "value"),
    )
    def _show_rows(count):
        """Reveal as many selector rows as asked for.

        The rows all exist from the start and are hidden rather than created,
        because Dash fixes its callback graph at startup -- a row added later
        would have no callbacks attached to it.
        """
        count = int(count or 1)
        base = {"gap": "0.5rem", "alignItems": "center", "marginBottom": "0.3rem"}
        return [
            {**base, "display": "flex" if i < count else "none"}
            for i in range(MAX_SERIES)
        ]

    @app.callback(
        Output(ID(PREFIX, "window-label"), "children"),
        Output(ID(PREFIX, "window"), "placeholder"),
        Output(ID(PREFIX, "window"), "value"),
        Input(ID(PREFIX, "mode"), "value"),
        State(ID(PREFIX, "window"), "value"),
    )
    def _window_for_mode(mode, current):
        """Relabel the window field when the x axis changes.

        The value resets to that mode's default rather than carrying over:
        365 is a sensible number of days and an unhelpful number of runs, and
        silently reinterpreting the same number in a different unit is how a
        plot ends up showing something other than what was asked for. A
        deliberately blank field is preserved -- "everything" means the same
        in both modes.
        """
        spec = MODES.get(mode, MODES["time"])
        if current is not None and str(current).strip() == "":
            return spec["label"], spec["default"], ""
        return spec["label"], spec["default"], spec["default"]

    @app.callback(
        Output(ID(PREFIX, "tick"), "disabled"),
        Input(ID(PREFIX, "auto"), "value"),
    )
    def _arm_refresh(auto):
        """Run the timer only while the box is ticked."""
        return "auto" not in (auto or [])

    @app.callback(
        Output(ID(PREFIX, "graph"), "figure"),
        Output(ID(PREFIX, "status"), "children"),
        Output(ID(PREFIX, "links"), "children"),
        Input({"type": PREFIX + "sid", "index": ALL}, "value"),
        Input({"type": PREFIX + "vid", "index": ALL}, "value"),
        Input(ID(PREFIX, "count"), "value"),
        Input(ID(PREFIX, "mode"), "value"),
        Input(ID(PREFIX, "window"), "value"),
        Input(ID(PREFIX, "opts"), "value"),
        # Only to make the timer redraw: its value is never read.  The
        # measurements are not cached, so re-running this callback is what
        # fetches current data.
        Input(ID(PREFIX, "tick"), "n_intervals"),
    )
    def _update(sids, vids, count, mode, window, opts, _tick):
        opts = opts or []
        mode = mode if mode in MODES else "time"
        count = int(count or 1)

        # Hidden rows keep whatever was last selected in them, so only the
        # visible ones count -- otherwise reducing the row count would leave
        # invisible series on the plot.
        pairs: List[Tuple[int, int]] = []
        for i in range(min(count, len(sids), len(vids))):
            if sids[i] is not None and vids[i] is not None:
                pairs.append((int(sids[i]), int(vids[i])))

        if not pairs:
            return {}, error_box("Choose a source and a metric."), ""

        # Changing a source fires this callback and the metric one at the same
        # time, so a row can arrive holding the previous source's metric.
        # Querying that pair returns nothing and would be reported as a
        # confirmed absence -- the wrong thing to say about a pairing that was
        # never valid. The answer is cached, so the check costs nothing.
        if not all(source_reports(s, v) for s, v in pairs):
            return dash.no_update, dash.no_update, dash.no_update

        window, complaint = _parse_window(window, mode)
        if complaint:
            # Leave the plot alone rather than blanking it: the user is
            # mid-edit, and an empty graph suggests the data vanished.
            return dash.no_update, error_box(complaint, kind="info"), dash.no_update

        notes: List[str] = []
        r = reader()

        if mode == "run":
            unusable = [s_ for s_, _ in pairs if not has_run_structure(s_)]
            if unusable:
                return (
                    {},
                    error_box(
                        "A selected source does not use run numbers -- every "
                        "interval is filed under run 0, because it is organised "
                        "by date. Switch the x axis back to time, or choose a "
                        "run-based source.",
                        kind="warn",
                    ),
                    "",
                )

        # One window for every panel. A run floor taken per panel would give
        # each its own x range, and panels that do not line up are not a
        # comparison.
        run_floor = None
        if mode == "run" and window is not None:
            all_runs = sorted(
                {run for s_, v_ in pairs for run in runs_for_metric(s_, v_)},
                reverse=True,
            )
            if all_runs:
                wanted = int(window)
                run_floor = all_runs[min(wanted, len(all_runs)) - 1]
                if wanted > len(all_runs):
                    notes.append(f"only {len(all_runs)} runs have these metrics")

        frames = []
        for sid, vid in pairs:
            query = dict(sid=sid, vid=vid, limit=None)
            if mode == "run":
                if run_floor is not None:
                    query["run_min"] = run_floor
            elif window is not None:
                query["recent_days"] = window
            try:
                frames.append(r.numbers(**query))
            except Exception as exc:
                LOGGER.exception("timeline query failed for sid=%s vid=%s", sid, vid)
                return {}, error_box(f"{type(exc).__name__}: {exc}"), ""

        if all(f.empty for f in frames):
            # The reader pushes filters into the query, so an empty result is
            # authoritative rather than an artifact of a scan window.
            return (
                {},
                error_box("No data for these selections in that window. "
                          "This is a real absence, not a truncated query.",
                          kind="info"),
                "",
            )

        name_by_vid = {
            int(v.vid): f"{v.groupx}/{v.subgroup}/{v.namex}"
            for v in r.values(vid=[v for _, v in pairs], limit=None).itertuples()
        }
        source_by_sid = {
            int(s_.sid): f"{s_.process}/{s_.stream} v{s_.version}"
            for s_ in r.sources(sid=[s_ for s_, _ in pairs], limit=None).itertuples()
        }

        # One spec per panel. Each is a complete figure in its own right, so
        # the same objects export as a stacked image without rebuilding.
        specs: List[PlotSpec] = []
        titles: List[str] = []
        total = 0
        code_totals: Dict[int, int] = {}

        for index, ((sid, vid), df) in enumerate(zip(pairs, frames)):
            metric = name_by_vid.get(vid, str(vid))
            source = source_by_sid.get(sid, str(sid))
            titles.append(f"{metric}  -  {source}")

            spec = PlotSpec(xlabel=MODES[mode]["axis"], ylabel=metric,
                            logy="logy" in opts, legend=False)
            if df.empty:
                notes.append(f"{metric}: no data")
                specs.append(spec)
                continue

            total += len(df)
            for k, n in df["code"].value_counts().to_dict().items():
                code_totals[int(k)] = code_totals.get(int(k), 0) + int(n)

            key = "start_run" if mode == "run" else "end_time"
            ordered = df.sort_values(key)
            # Colour by row so a panel matches the swatch beside its controls.
            spec.add_series(ordered[key], ordered["valuex"],
                            yerr=ordered["sigma"], label=metric,
                            color=PALETTE[index % len(PALETTE)],
                            style="markers")

            if "limits" in opts:
                # Each panel has its own y axis, so its own band is readable
                # here in a way it was not when everything shared one scale.
                try:
                    # Every band for this metric, not just the ones near the
                    # window: a band commited long ago still governs today.
                    # dqm.bands resolves which one applies where.
                    bands = newest_first(r.limits(sid=sid, vid=vid, limit=None))
                    lo, hi = _band_steps(ordered, key, vid, bands)
                    if lo:
                        # dqm.limits is populated independently of dqm.numbers,
                        # so a band may be absent even where measurements exist.
                        spec.add_series(*zip(*lo), label="llimit",
                                        color="#d62728", style="lines")
                        spec.add_series(*zip(*hi), label="ulimit",
                                        color="#d62728", style="lines")
                except Exception:
                    LOGGER.debug("no limits for sid=%s vid=%s", sid, vid,
                                 exc_info=True)

            specs.append(spec)

        code_text = ", ".join(
            f"{CODE_VALUES.get(k, k)}: {n}" for k, n in sorted(code_totals.items())
        )
        if mode == "run":
            shown = sorted({int(x) for f in frames if not f.empty
                            for x in f["start_run"].dropna()})
            span = (f"runs {shown[0]} to {shown[-1]} ({len(shown)} runs)"
                    if shown else "no runs")
        else:
            lo = min(f["end_time"].min() for f in frames if not f.empty)
            hi = max(f["end_time"].max() for f in frames if not f.empty)
            span = f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}"
        if window is None:
            notes.append(f"showing {MODES[mode]['blank']}")

        drawn = sum(1 for sp in specs if sp.series)
        status = html.Div(
            [
                html.Span(f"{total} points", style={"fontWeight": 600}),
                html.Span(f" in {drawn} plot{'s' if drawn != 1 else ''}",
                          style={"color": "#555"}),
                html.Span(f"  |  {code_text}", style={"color": "#555"}),
                html.Span(f"  |  {span}", style={"color": "#555"}),
                *([html.Span(f"  |  {'; '.join(notes)}", style={"color": "#8a6d00"})]
                  if notes else []),
                # When the plot was drawn.  Without it there is no way to tell
                # a working auto-refresh from a stalled one, since a refresh
                # that returns the same data looks exactly like no refresh.
                # UTC, to match the time axis.
                html.Span("  |  " + drawn_stamp(), style={"color": "#888"}),
            ]
        )

        # The same figure as a static image, for a note or an elog post --
        # rendered from this same spec by dqm.plot.static.
        series_arg = ",".join(f"{s}:{v}" for s, v in pairs)
        qs = f"?series={series_arg}&mode={mode}"
        if window is not None:
            qs += f"&window={window:g}"
        links = html.Div(
            [
                html.Span("report image: ", style={"color": "#555"}),
                *[
                    html.A(ext.upper(), href=_report_href(ext, qs),
                           target="_blank", style={"marginRight": "0.75rem"})
                    for ext in ("png", "pdf", "svg")
                ],
            ]
        )
        return (to_plotly_grid(specs, shared_x=True, titles=titles),
                status, links)
