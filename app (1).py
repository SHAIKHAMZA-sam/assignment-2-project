"""
🎓 Student Marks Analytics Dashboard
====================================
Interactive visualization of student marks (percentages) across subjects.

Run with:  streamlit run app.py

Works with any CSV that has Student_ID, Student_Name, Gender, Section and one
numeric column per subject (marks out of 100). Extra columns such as
Attendance_% and Study_Hours_Per_Day are used when present.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Student Marks Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "student_marks_dataset.csv"

ID, NAME, GENDER, SECTION = "Student_ID", "Student_Name", "Gender", "Section"
ATT, STUDY = "Attendance_%", "Study_Hours_Per_Day"
REQUIRED = [ID, NAME, GENDER, SECTION]
DERIVED = ["Total_Marks", "Percentage", "Subjects_Failed", "Result", "Grade"]
NON_SUBJECT = set(REQUIRED + [ATT, STUDY] + DERIVED)

GRADE_ORDER = ["A+", "A", "B+", "B", "C", "D", "F"]
COLORWAY = ["#2F6FDE", "#E8833A", "#1FA67A", "#C2457B", "#7A5CD6", "#D4A017", "#2AA3B8", "#8A94A6"]
GOOD, BAD, NEUTRAL = "#1FA67A", "#D9534F", "#9AA4B2"
px.defaults.color_discrete_sequence = COLORWAY

PAGES = [
    "🏠 Overview",
    "📚 Subject Analysis",
    "🧑‍🎓 Student Profile",
    "⚖️ Compare",
    "🔗 Correlations",
    "🧾 Data Explorer",
]

try:
    _V = tuple(int(p) for p in st.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    _V = (1, 40)
NEW_WIDTH_API = _V >= (1, 50)  # Streamlit ≥ 1.50 replaced use_container_width


# ──────────────────────────────────────────────────────────────────────────────
# Styling and small helpers
# ──────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
        h1 { font-weight: 650; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
        h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
        .app-subtitle { font-size: 1.02rem; opacity: 0.72; margin: 0 0 1.1rem 0; }
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.22);
            border-left: 3px solid #2F6FDE; border-radius: 8px; padding: 0.7rem 0.9rem;
        }
        div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-weight: 600; }
        .insight { border-left: 3px solid #1FA67A; padding: 0.35rem 0.8rem; margin: 0.2rem 0 0.8rem 0;
                   background: rgba(31,166,122,0.08); border-radius: 0 6px 6px 0; font-size: 0.93rem; }
        .footer-note { font-size: 0.85rem; opacity: 0.7; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def insight(text: str) -> None:
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def ok(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def pct(x, digits: int = 1) -> str:
    return f"{float(x):.{digits}f}%" if ok(x) else "—"


PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig: go.Figure, key: str | None = None) -> None:
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


def show_table(df, height: int | None = None, **kwargs) -> None:
    kwargs.setdefault("hide_index", True)
    if height is not None:
        kwargs["height"] = height
    if NEW_WIDTH_API:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def style(fig: go.Figure, title: str, x: str | None = None, y: str | None = None,
          height: int = 400, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        height=height, margin=dict(l=10, r=10, t=56, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    if x is not None:
        fig.update_xaxes(title_text=x)
    if y is not None:
        fig.update_yaxes(title_text=y)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Data loading and preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate columns, clean types and add Total_Marks / Percentage. Returns (df, subjects)."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required column(s): " + ", ".join(missing)
            + ". Expected at least: Student_ID, Student_Name, Gender, Section and one column per subject."
        )

    candidates = [c for c in df.columns if c not in NON_SUBJECT]
    subjects = []
    for c in candidates:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() >= 0.8:  # mostly numeric → a subject column
            df[c] = num.clip(0, 100)
            subjects.append(c)
    if len(subjects) < 2:
        raise ValueError("Could not find at least two numeric subject columns (marks out of 100).")

    for c in [ID, NAME, GENDER, SECTION]:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    for c in [ATT, STUDY]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=subjects, how="all").drop_duplicates(subset=[ID], keep="first")
    df["Total_Marks"] = df[subjects].sum(axis=1, skipna=True)
    df["Percentage"] = df[subjects].mean(axis=1, skipna=True).round(2)  # each subject is out of 100
    return df.reset_index(drop=True), subjects


@st.cache_data(show_spinner="Loading dataset…")
def load_path(path: str, mtime: float):
    return prepare(pd.read_csv(path))


@st.cache_data(show_spinner="Reading the uploaded file…")
def load_bytes(raw: bytes):
    return prepare(pd.read_csv(io.BytesIO(raw)))


def get_dataset():
    error, result = None, None
    if DATA_PATH.exists():
        try:
            result = load_path(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Could not read {DATA_PATH.name}: {exc}"
    else:
        st.error(f"Dataset not found. Put **student_marks_dataset.csv** in `{DATA_PATH.parent}` or upload it below.")
        up = st.file_uploader("Upload student marks CSV", type="csv")
        if up is None:
            st.stop()
        try:
            result = load_bytes(up.getvalue())
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Could not read the uploaded file: {exc}"
    if error:
        st.error(error)
        st.stop()
    if result is None or result[0].empty:
        st.error("The dataset has no usable rows after cleaning.")
        st.stop()
    return result


def grade_of(p: float, failed: int) -> str:
    if failed > 0:
        return "F"
    return "A+" if p >= 90 else "A" if p >= 80 else "B+" if p >= 70 else "B" if p >= 60 else "C" if p >= 50 else "D"


def derive(df: pd.DataFrame, subjects: list[str], pass_mark: int) -> pd.DataFrame:
    """Result / Subjects_Failed / Grade depend on the pass mark chosen in the sidebar."""
    out = df.copy()
    out["Subjects_Failed"] = (out[subjects] < pass_mark).sum(axis=1)
    out["Result"] = np.where(out["Subjects_Failed"] == 0, "Pass", "Fail")
    out["Grade"] = [grade_of(p, int(n)) for p, n in zip(out["Percentage"], out["Subjects_Failed"])]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def build_sidebar(df: pd.DataFrame, subjects: list[str]):
    sb = st.sidebar
    sb.markdown("### 🎓 Student Analytics")
    page = sb.radio("Navigate", PAGES, label_visibility="collapsed", key="nav_page")
    sb.divider()

    with sb.expander("Advanced controls"):
        pass_mark = st.slider("Pass mark (per subject)", 0, 60, 35, key="pass_mark",
                              help="A student fails if any subject is below this mark.")

    d = derive(df, subjects, pass_mark)
    sb.markdown("#### Filters")
    sections = sb.multiselect("Section", sorted(d[SECTION].unique()), key="f_section")
    genders = sb.multiselect("Gender", sorted(d[GENDER].unique()), key="f_gender")
    results = sb.multiselect("Result", ["Pass", "Fail"], key="f_result")
    lo, hi = int(np.floor(d["Percentage"].min())), int(np.ceil(d["Percentage"].max()))
    pct_rng = sb.slider("Percentage range", lo, hi, (lo, hi), key="f_pct") if lo < hi else (lo, hi)
    att_rng = None
    if ATT in d.columns and d[ATT].notna().any():
        a_lo, a_hi = int(np.floor(d[ATT].min())), int(np.ceil(d[ATT].max()))
        if a_lo < a_hi:
            att_rng = sb.slider("Attendance (%)", a_lo, a_hi, (a_lo, a_hi), key="f_att")

    mask = d["Percentage"].between(pct_rng[0], pct_rng[1])
    if sections:
        mask &= d[SECTION].isin(sections)
    if genders:
        mask &= d[GENDER].isin(genders)
    if results:
        mask &= d["Result"].isin(results)
    if att_rng is not None:
        mask &= d[ATT].between(att_rng[0], att_rng[1]) | d[ATT].isna()
    f = d[mask]

    sb.divider()
    sb.caption(f"**{len(f):,}** of **{len(d):,}** students match · {len(subjects)} subjects · pass mark {pass_mark}")
    return page, d, f, pass_mark


# ──────────────────────────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────────────────────────
def render_kpis(f: pd.DataFrame, subjects: list[str]) -> None:
    means = f[subjects].mean()
    top = f.loc[f["Percentage"].idxmax()]
    low = f.loc[f["Percentage"].idxmin()]
    c = st.columns(6)
    c[0].metric("Students", f"{len(f):,}")
    c[1].metric("Average percentage", pct(f["Percentage"].mean()))
    c[2].metric("Highest percentage", pct(top["Percentage"]), help=f"{top[NAME]} ({top[ID]})")
    c[3].metric("Lowest percentage", pct(low["Percentage"]), help=f"{low[NAME]} ({low[ID]})")
    c[4].metric("Pass rate", pct((f["Result"] == "Pass").mean() * 100))
    if ATT in f.columns and f[ATT].notna().any():
        c[5].metric("Average attendance", pct(f[ATT].mean()))
    else:
        c[5].metric("Top subject", str(means.idxmax()), help=f"Average {means.max():.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
def page_overview(f: pd.DataFrame, subjects: list[str], pass_mark: int) -> None:
    means = f[subjects].mean()
    best, worst = means.idxmax(), means.idxmin()
    insight(f"<b>{best}</b> has the highest average ({means[best]:.1f}) and <b>{worst}</b> the lowest ({means[worst]:.1f}). "
            f"{(f['Result'] == 'Fail').sum():,} of {len(f):,} students fail at least one subject.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=means.index, y=means.values, marker_color=COLORWAY[0],
            text=[f"{v:.1f}" for v in means.values], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Average: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=pass_mark, line_dash="dash", line_color=BAD, annotation_text=f"pass mark {pass_mark}")
        fig.update_yaxes(range=[0, 100])
        show(style(fig, "Average marks per subject", "Subject", "Average marks", legend=False), key="ov_avg")
    with c2:
        fig = px.histogram(f, x="Percentage", nbins=20)
        fig.update_traces(marker_color=COLORWAY[1], marker_line_width=0.5, marker_line_color="rgba(0,0,0,0.25)",
                          hovertemplate="Percentage: %{x}<br>Students: %{y}<extra></extra>")
        fig.add_vline(x=float(f["Percentage"].mean()), line_dash="dash", line_color=NEUTRAL,
                      annotation_text=f"mean {f['Percentage'].mean():.1f}%")
        show(style(fig, "Distribution of overall percentage", "Overall percentage", "Number of students",
                   legend=False), key="ov_hist")

    c3, c4 = st.columns(2)
    with c3:
        counts = f["Grade"].value_counts().reindex(GRADE_ORDER).dropna()
        colors = [BAD if g == "F" else COLORWAY[i % len(COLORWAY)] for i, g in enumerate(counts.index)]
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.5, sort=False,
                               marker=dict(colors=colors), textinfo="label+percent",
                               hovertemplate="<b>Grade %{label}</b><br>%{value} students (%{percent})<extra></extra>"))
        show(style(fig, "Grade distribution", height=400), key="ov_grade")
    with c4:
        g = f.groupby([SECTION, "Result"]).size().reset_index(name="Students")
        fig = px.bar(g, x=SECTION, y="Students", color="Result", barmode="group",
                     color_discrete_map={"Pass": GOOD, "Fail": BAD})
        show(style(fig, "Pass and fail by section", "Section", "Students"), key="ov_section")

    st.markdown("### Top and bottom performers")
    n = st.radio("List length", [5, 10, 15], index=1, horizontal=True, key="ov_n", format_func=lambda x: f"Top {x}")
    cols = [ID, NAME, SECTION, "Percentage", "Grade"]
    l, r = st.columns(2)
    with l:
        top = f.nlargest(n, "Percentage")[cols]
        fig = go.Figure(go.Bar(x=top["Percentage"], y=top[NAME] + " (" + top[ID] + ")", orientation="h",
                               marker_color=GOOD, hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, f"Top {n} students", "Percentage", None, height=max(300, 28 * n + 90), legend=False), key="ov_top")
    with r:
        bottom = f.nsmallest(n, "Percentage")[cols]
        fig = go.Figure(go.Bar(x=bottom["Percentage"], y=bottom[NAME] + " (" + bottom[ID] + ")", orientation="h",
                               marker_color=BAD, hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, f"Bottom {n} students — may need support", "Percentage", None,
                   height=max(300, 28 * n + 90), legend=False), key="ov_bottom")


def page_subjects(f: pd.DataFrame, subjects: list[str], pass_mark: int) -> None:
    st.subheader("📚 Subject analysis")
    t1, t2, t3, t4, t5 = st.tabs(["Averages and pass rate", "Distribution", "Single subject", "By group", "Section heatmap"])

    with t1:
        stats = pd.DataFrame({
            "Subject": subjects,
            "Average": [f[s].mean() for s in subjects],
            "Median": [f[s].median() for s in subjects],
            "Highest": [f[s].max() for s in subjects],
            "Lowest": [f[s].min() for s in subjects],
            "Std deviation": [f[s].std() for s in subjects],
            "Pass rate %": [(f[s] >= pass_mark).mean() * 100 for s in subjects],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=stats["Subject"], y=stats["Average"], name="Average", marker_color=COLORWAY[0]))
        fig.add_trace(go.Scatter(x=stats["Subject"], y=stats["Pass rate %"], name="Pass rate %", mode="lines+markers",
                                 line=dict(color=COLORWAY[1], width=2.4)))
        fig.update_yaxes(range=[0, 105])
        show(style(fig, "Average marks and pass rate by subject", "Subject", "Marks / pass rate (%)"), key="sa_avg")
        show_table(stats.style.format({c: "{:.1f}" for c in stats.columns if c != "Subject"}))
        hardest = stats.loc[stats["Pass rate %"].idxmin()]
        insight(f"<b>{hardest['Subject']}</b> has the lowest pass rate ({hardest['Pass rate %']:.1f}%).")

    with t2:
        fig = go.Figure()
        for i, s in enumerate(subjects):
            fig.add_trace(go.Box(y=f[s], name=s, boxpoints="outliers", marker_color=COLORWAY[i % len(COLORWAY)]))
        fig.add_hline(y=pass_mark, line_dash="dash", line_color=BAD)
        show(style(fig, "Marks distribution by subject (box plot — dots are outliers)", "Subject", "Marks",
                   height=480, legend=False), key="sa_box")

    with t3:
        subj = st.selectbox("Subject", subjects, key="sa_subject")
        v = f[subj].dropna()
        m = st.columns(4)
        m[0].metric("Average", f"{v.mean():.1f}")
        m[1].metric("Highest", f"{v.max():.0f}")
        m[2].metric("Lowest", f"{v.min():.0f}")
        m[3].metric("Pass rate", pct((v >= pass_mark).mean() * 100))
        fig = px.histogram(f, x=subj, nbins=20, color=GENDER, barmode="overlay", opacity=0.7)
        fig.add_vline(x=pass_mark, line_dash="dash", line_color=BAD, annotation_text="pass mark")
        show(style(fig, f"{subj} — marks distribution by gender", "Marks", "Number of students"), key="sa_single")
        top = f.nlargest(5, subj)[[ID, NAME, SECTION, subj]]
        st.markdown(f"**Top 5 in {subj}**")
        show_table(top)

    with t4:
        grp = st.radio("Group by", [c for c in [SECTION, GENDER, "Grade", "Result"] if c in f.columns],
                       horizontal=True, key="sa_group")
        means = f.groupby(grp)[subjects].mean().reset_index().melt(id_vars=grp, var_name="Subject", value_name="Average")
        fig = px.bar(means, x="Subject", y="Average", color=grp, barmode="group")
        fig.update_yaxes(range=[0, 100])
        show(style(fig, f"Average marks by subject and {grp.lower()}", "Subject", "Average marks"), key="sa_group_fig")

    with t5:
        piv = f.groupby(SECTION)[subjects].mean()
        fig = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns, y=piv.index, colorscale="RdYlGn", text=np.round(piv.values, 1),
            texttemplate="%{text}", hovertemplate="Section %{y} · %{x}<br>Average: %{z:.1f}<extra></extra>",
            colorbar=dict(title=dict(text="Avg")),
        ))
        fig.update_yaxes(autorange="reversed", type="category")
        show(style(fig, "Average marks — section × subject", "Subject", "Section", height=360, legend=False),
             key="sa_heat")


def radar(traces: list[tuple[str, list[float], str]], labels: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for name, vals, color in traces:
        fig.add_trace(go.Scatterpolar(
            r=list(vals) + [vals[0]], theta=labels + [labels[0]], fill="toself", name=name, opacity=0.55,
            line=dict(color=color, width=2),
            hovertemplate="%{theta}: %{r:.1f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return style(fig, title, height=520)


def page_profile(f: pd.DataFrame, subjects: list[str]) -> None:
    st.subheader("🧑‍🎓 Student profile")
    people = f.sort_values(NAME)
    labels = (people[NAME] + " — " + people[ID]).tolist()
    pick = st.selectbox("Student", labels, key=f"sp_pick_{len(labels)}_{hash(tuple(labels[:5]))}",
                        help="Type a name or ID to search.")
    row = people.iloc[labels.index(pick)]
    rank = int((f["Percentage"] > row["Percentage"]).sum()) + 1

    c = st.columns(5)
    c[0].metric("Percentage", pct(row["Percentage"], 2))
    c[1].metric("Rank", f"{rank} of {len(f)}", help="Within the students matching the current filters.")
    c[2].metric("Grade", str(row["Grade"]))
    c[3].metric("Result", str(row["Result"]))
    if ATT in f.columns and ok(row.get(ATT)):
        c[4].metric("Attendance", pct(row[ATT]))
    else:
        c[4].metric("Section", str(row[SECTION]))

    marks = row[subjects].astype(float)
    avg = f[subjects].mean()
    best, weak = marks.idxmax(), marks.idxmin()
    insight(f"<b>{row[NAME]}</b> is strongest in <b>{best}</b> ({marks[best]:.0f}) and weakest in <b>{weak}</b> ({marks[weak]:.0f}). "
            f"Overall they are {abs(row['Percentage'] - f['Percentage'].mean()):.1f} points "
            f"{'above' if row['Percentage'] >= f['Percentage'].mean() else 'below'} the average of the current selection.")

    l, r = st.columns(2)
    with l:
        show(radar([(row[NAME], marks.tolist(), COLORWAY[0]), ("Class average", avg.tolist(), NEUTRAL)],
                   subjects, "Subject profile vs class average"), key="sp_radar")
    with r:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=subjects, y=marks.values, name=row[NAME], marker_color=COLORWAY[0]))
        fig.add_trace(go.Bar(x=subjects, y=avg.values, name="Class average", marker_color=NEUTRAL))
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 100])
        show(style(fig, "Marks by subject", "Subject", "Marks", height=520), key="sp_bar")


def page_compare(f: pd.DataFrame, subjects: list[str]) -> None:
    st.subheader("⚖️ Compare")
    t1, t2 = st.tabs(["Compare groups", "Compare students"])

    with t1:
        grp = st.radio("Compare by", [SECTION, GENDER, "Grade", "Result"], horizontal=True, key="cmp_group")
        means = f.groupby(grp)[subjects].mean()
        if len(means) < 2:
            st.warning(f"Only one {grp.lower()} in the current filter. Widen the filters to compare groups.")
        else:
            traces = [(f"{grp} {k}", means.loc[k].tolist(), COLORWAY[i % len(COLORWAY)]) for i, k in enumerate(means.index[:8])]
            show(radar(traces, subjects, f"Average marks by {grp.lower()}"), key="cmp_group_radar")
            show_table(means.round(1).reset_index())

    with t2:
        people = f.sort_values(NAME)
        labels = (people[NAME] + " — " + people[ID]).tolist()
        chosen = st.multiselect("Students (2–6)", labels, default=labels[:3],
                                key=f"cmp_students_{len(labels)}_{hash(tuple(labels[:3]))}")
        if len(chosen) < 2:
            st.warning("Select at least **two** students to compare.")
        else:
            rows = people.iloc[[labels.index(c) for c in chosen]]
            traces = [(r[NAME], r[subjects].astype(float).tolist(), COLORWAY[i % len(COLORWAY)])
                      for i, (_, r) in enumerate(rows.iterrows())]
            show(radar(traces, subjects, "Student comparison radar"), key="cmp_students_radar")
            show_table(rows[[ID, NAME, SECTION] + subjects + ["Percentage", "Grade"]])


def page_correlations(f: pd.DataFrame, subjects: list[str]) -> None:
    st.subheader("🔗 Correlations")
    numeric = subjects + [c for c in [ATT, STUDY] if c in f.columns and f[c].notna().any()] + ["Percentage"]
    if len(f) < 3:
        st.warning("Need at least 3 students to compute correlations. Widen the filters.")
        return

    corr = f[numeric].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>", colorbar=dict(title=dict(text="Corr.")),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    show(style(fig, "Correlation heatmap — subjects, habits and overall percentage", height=560, legend=False),
         key="co_heat")
    if "Percentage" in corr.columns:
        pc = corr["Percentage"].drop("Percentage").dropna()
        if len(pc):
            insight(f"<b>{pc.idxmax()}</b> is most strongly linked to overall percentage (r = {pc.max():.2f}). "
                    "Correlation shows association, not cause.")

    st.markdown("### Scatter explorer")
    c1, c2, c3 = st.columns(3)
    default_x = STUDY if STUDY in numeric else numeric[0]
    xcol = c1.selectbox("X-axis", numeric, index=numeric.index(default_x), key="co_x")
    ycol = c2.selectbox("Y-axis", numeric, index=numeric.index("Percentage"), key="co_y")
    color = c3.selectbox("Colour by", [GENDER, SECTION, "Result", "Grade"], key="co_color")
    plot = f.dropna(subset=[xcol, ycol])
    if plot.empty:
        st.warning("No rows with valid values for these axes.")
        return
    fig = px.scatter(plot, x=xcol, y=ycol, color=color, opacity=0.75, custom_data=[NAME, ID, SECTION, "Percentage"])
    fig.update_traces(marker=dict(size=8), hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Section %{customdata[2]}"
                      f"<br>{xcol}: " "%{x}" f"<br>{ycol}: " "%{y}<br>Overall: %{customdata[3]:.1f}%<extra></extra>")
    if xcol != ycol and plot[xcol].nunique() > 2:  # simple least-squares trend line
        slope, intercept = np.polyfit(plot[xcol].astype(float), plot[ycol].astype(float), 1)
        xs = np.array([plot[xcol].min(), plot[xcol].max()], dtype=float)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Trend",
                                 line=dict(color=NEUTRAL, dash="dash", width=2)))
    show(style(fig, f"{ycol} vs {xcol}", xcol, ycol, height=500), key="co_scatter")


def page_explorer(f: pd.DataFrame, subjects: list[str]) -> None:
    st.subheader("🧾 Data Explorer")
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    search = c1.text_input("Search by name or ID", placeholder="e.g. Priya or STU1042", key="ex_search").strip()
    sort_col = c2.selectbox("Sort by", list(f.columns), index=list(f.columns).index("Percentage"), key="ex_sort")
    order = c3.radio("Order", ["Descending", "Ascending"], key="ex_order")
    n_rows = c4.selectbox("Rows", [50, 100, 250, 500, 1000], index=3, key="ex_rows")

    view = f
    if search:
        hit = view[NAME].str.contains(search, case=False, regex=False) | view[ID].str.contains(search, case=False, regex=False)
        view = view[hit]
    if view.empty:
        st.warning("No students match your search and filters.")
        return
    shown = view.sort_values(sort_col, ascending=(order == "Ascending"), kind="stable").head(int(n_rows))
    st.caption(f"Showing **{len(shown):,}** of **{len(view):,}** matching students.")
    show_table(shown, height=520)

    with st.expander("Summary statistics"):
        show_table(view[subjects + ["Total_Marks", "Percentage"]].describe().T.round(2).reset_index()
                   .rename(columns={"index": "column"}))

    st.download_button("⬇️ Download Filtered Data", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_student_marks.csv", mime="text/csv", type="primary")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    st.title("🎓 Student Marks Analytics Dashboard")
    st.markdown('<p class="app-subtitle">Interactive analysis of student performance across subjects, sections and study habits</p>',
                unsafe_allow_html=True)

    df, subjects = get_dataset()
    page, d, f, pass_mark = build_sidebar(df, subjects)
    if f.empty:
        st.warning("No students match the current filters. Clear a filter or widen the percentage range.")
        st.stop()

    render_kpis(f, subjects)
    st.markdown("")

    if page == PAGES[0]:
        page_overview(f, subjects, pass_mark)
    elif page == PAGES[1]:
        page_subjects(f, subjects, pass_mark)
    elif page == PAGES[2]:
        page_profile(f, subjects)
    elif page == PAGES[3]:
        page_compare(f, subjects)
    elif page == PAGES[4]:
        page_correlations(f, subjects)
    else:
        page_explorer(f, subjects)

    st.divider()
    st.markdown('<p class="footer-note"><em>This dashboard is for educational and analytical purposes only.</em></p>',
                unsafe_allow_html=True)


main()
