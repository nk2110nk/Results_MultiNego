#!/usr/bin/env python3
"""Aggregate all negotiation results and reproduce the requested figures/tables."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent

KNOWN_DOMAINS = (
    "Laptop",
    "ItexvsCypress",
    "IS_BT_Acquisition",
    "Grocery",
    "thompson",
    "Car",
    "EnergySmall_A",
)
UNKNOWN_DOMAINS = ("Coffee", "Camera", "Lunch", "SmartPhone", "Kitchen")
COMMON_UNKNOWN_DOMAINS = ("Camera", "Lunch", "Kitchen")
AGENTS = ("Boulware", "Conceder", "Linear", "Atlas3")
PAIRS = tuple(
    f"{left}-{right}"
    for left_index, left in enumerate(AGENTS)
    for right in AGENTS[left_index:]
)


@dataclass(frozen=True)
class Series:
    name: str
    root: str
    mode: str
    cases: tuple[str, ...]
    color: str


CASES_1_TO_6 = tuple(f"case{index}" for index in range(1, 7))
SERIES = (
    Series("Transformer-base_general", "results_transformer-based", "general", CASES_1_TO_6, "#008837"),
    Series("Transformer-base_expert", "results_transformer-based", "expert", CASES_1_TO_6, "#984EA3"),
    Series("MiPN-base_general", "results_MiPN-based", "general", CASES_1_TO_6, "#FFD700"),
    Series("MiPN-base_expert", "results_MiPN-based", "expert", CASES_1_TO_6, "#E41A1C"),
    Series("RLBOA-base_general", "results_RLBOA-based", "general", CASES_1_TO_6, "#FF7F00"),
    Series("RLBOA-base_expert", "results_RLBOA-based", "expert", CASES_1_TO_6, "#0072B2"),
    Series("α-Nego-base_general", "results_α-Nego-based", "general", ("case1",), "#00A6D6"),
    Series("α-Nego-base_expert", "results_α-Nego-based", "expert", ("case1",), "#4D4D4D"),
)


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    filename: str
    percent: bool = False


METRICS = (
    Metric("my_util", "Individual Utility", "individual_utility"),
    Metric("opp_util1", "Opponent 1 Utility", "opponent1_utility"),
    Metric("opp_util2", "Opponent 2 Utility", "opponent2_utility"),
    Metric("social", "Social Welfare", "social_welfare"),
    Metric("nash", "Nash Product", "nash_product"),
    Metric("step", "Number of Steps", "number_of_steps"),
    Metric("step_efficiency", "Step Efficiency", "step_efficiency"),
    Metric("agreement_rate", "Agreement Rate", "agreement_rate", percent=True),
)
METRIC_BY_KEY = {metric.key: metric for metric in METRICS}
OVERVIEW_METRICS = ("my_util", "step", "step_efficiency", "agreement_rate")
TABLE_METRIC_GROUPS = (
    ("individual_utility_agreement_rate", ("my_util", "agreement_rate")),
    ("opponent_utilities", ("opp_util1", "opp_util2")),
    ("social_welfare_nash_product", ("social", "nash")),
    ("steps_step_efficiency", ("step", "step_efficiency")),
)
DOMAIN_META = {
    "Laptop": ("Manufacturing", "27"),
    "ItexvsCypress": ("SCM", "180"),
    "IS_BT_Acquisition": ("Buyout", "384"),
    "Grocery": ("Retail", "1600"),
    "thompson": ("Employment", "3125"),
    "Car": ("Mobility", "15625"),
    "EnergySmall_A": ("Energy Plant", "15625"),
    "Coffee": ("Unseen", "112"),
    "Camera": ("Unseen", "3600"),
    "Lunch": ("Unseen", "3840"),
    "SmartPhone": ("Unseen", "12000"),
    "Kitchen": ("Unseen", "15625"),
}


@dataclass
class Accumulator:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_square += value * value

    def merge(self, other: "Accumulator") -> None:
        self.count += other.count
        self.total += other.total
        self.total_square += other.total_square

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    @property
    def std(self) -> float:
        if not self.count:
            return math.nan
        variance = max(0.0, self.total_square / self.count - self.mean * self.mean)
        return math.sqrt(variance)


CellKey = tuple[str, str, str, str]


def series_domains(series: Series) -> tuple[str, ...]:
    if series.mode == "expert":
        return KNOWN_DOMAINS
    if series.root == "results_α-Nego-based":
        return KNOWN_DOMAINS + COMMON_UNKNOWN_DOMAINS
    return KNOWN_DOMAINS + UNKNOWN_DOMAINS


def values_from_row(row: dict[str, str], path: Path) -> dict[str, float]:
    try:
        values = {
            key: float(row[key])
            for key in ("my_util", "opp_util1", "opp_util2", "social", "nash", "step")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric row in {path}: {exc}") from exc
    values["step_efficiency"] = values["my_util"] / values["step"] if values["step"] else 0.0
    values["agreement_rate"] = 100.0 if values["my_util"] != 0.0 else 0.0
    return values


def load_cells() -> dict[CellKey, Accumulator]:
    cells: dict[CellKey, Accumulator] = {}
    for series in SERIES:
        split_root = ROOT / series.root / series.mode
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        for pair in PAIRS:
            pair_dir = split_root / pair
            if not pair_dir.is_dir():
                left, right = pair.split("-", 1)
                reversed_dir = split_root / f"{right}-{left}"
                if reversed_dir.is_dir():
                    pair_dir = reversed_dir
            for domain in series_domains(series):
                for case_name in series.cases:
                    case_dir = pair_dir / domain / case_name
                    files = sorted(case_dir.glob("*.tsv"))
                    if len(files) != 1:
                        raise ValueError(f"Expected one TSV, found {len(files)}: {case_dir}")
                    path = files[0]
                    row_count = 0
                    with path.open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle, delimiter="\t")
                        for row in reader:
                            row_count += 1
                            for metric, value in values_from_row(row, path).items():
                                key = (series.name, domain, pair, metric)
                                cells.setdefault(key, Accumulator()).add(value)
                    if row_count != 100:
                        raise ValueError(f"Expected 100 rows, found {row_count}: {path}")
    return cells


def merged_stats(
    cells: dict[CellKey, Accumulator],
    series_name: str,
    domains: tuple[str, ...],
    pairs: tuple[str, ...],
    metric: str,
) -> Accumulator | None:
    result = Accumulator()
    for domain in domains:
        for pair in pairs:
            cell = cells.get((series_name, domain, pair, metric))
            if cell is not None:
                result.merge(cell)
    return result if result.count else None


def domain_stats(
    cells: dict[CellKey, Accumulator], group: str, metric: str, domain: str, series_name: str
) -> Accumulator | None:
    if domain == "Average":
        domains = KNOWN_DOMAINS if group == "known" else COMMON_UNKNOWN_DOMAINS
    else:
        domains = (domain,)
    return merged_stats(cells, series_name, domains, PAIRS, metric)


def opponent_stats(
    cells: dict[CellKey, Accumulator], group: str, metric: str, pair: str, series_name: str
) -> Accumulator | None:
    domains = KNOWN_DOMAINS if group == "known" else COMMON_UNKNOWN_DOMAINS
    return merged_stats(cells, series_name, domains, (pair,), metric)


def available_series(group: str) -> tuple[Series, ...]:
    return SERIES if group == "known" else tuple(series for series in SERIES if series.mode == "general")


def format_value(metric: Metric, stats: Accumulator | None) -> str:
    if stats is None:
        return "N/A"
    if metric.percent:
        return f"{stats.mean:.1f}%"
    if metric.key == "step":
        return f"{stats.mean:.2f}"
    return f"{stats.mean:.3f}"


def text_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header = "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers))
    rule = "  ".join("-" * width for width in widths)
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows]
    return "\n".join([header, rule, *body])


def write_domain_tables(cells: dict[CellKey, Accumulator], group: str, domains: tuple[str, ...]) -> None:
    table_dir = OUTPUT / "tables" / group / "by_domain"
    table_dir.mkdir(parents=True, exist_ok=True)
    combined: list[str] = []
    headers = ["Method", *domains, "Average"]
    for metric in METRICS:
        rows = []
        for series in SERIES:
            rows.append(
                [series.name]
                + [format_value(metric, domain_stats(cells, group, metric.key, domain, series.name)) for domain in domains]
                + [format_value(metric, domain_stats(cells, group, metric.key, "Average", series.name))]
            )
        rendered = f"{metric.label}\n{text_table(headers, rows)}"
        (table_dir / f"{metric.filename}.txt").write_text(rendered + "\n", encoding="utf-8")
        combined.append(rendered)
    (OUTPUT / "tables" / group / "domain_tables.txt").write_text(
        "\n\n".join(combined) + "\n", encoding="utf-8"
    )


def write_opponent_tables(cells: dict[CellKey, Accumulator], group: str) -> None:
    table_dir = OUTPUT / "tables" / group / "by_opponent"
    table_dir.mkdir(parents=True, exist_ok=True)
    pair_sections = (PAIRS[:5], PAIRS[5:])
    combined: list[str] = []
    for metric in METRICS:
        metric_sections = []
        for pairs in pair_sections:
            headers = ["Method", *pairs]
            rows = [
                [series.name]
                + [format_value(metric, opponent_stats(cells, group, metric.key, pair, series.name)) for pair in pairs]
                for series in SERIES
            ]
            metric_sections.append(text_table(headers, rows))
        rendered = f"{metric.label}\n" + "\n\n".join(metric_sections)
        (table_dir / f"{metric.filename}.txt").write_text(rendered + "\n", encoding="utf-8")
        combined.append(rendered)
    (OUTPUT / "tables" / group / "opponent_tables.txt").write_text(
        "\n\n".join(combined) + "\n", encoding="utf-8"
    )


def write_summary_csvs(cells: dict[CellKey, Accumulator], group: str, domains: tuple[str, ...]) -> None:
    data_dir = OUTPUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    domain_path = data_dir / f"{group}_by_domain.csv"
    with domain_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "domain", "series", "mean", "std", "count"])
        for metric in METRICS:
            for domain in (*domains, "Average"):
                for series in SERIES:
                    stats = domain_stats(cells, group, metric.key, domain, series.name)
                    writer.writerow(
                        [metric.key, domain, series.name]
                        + ([f"{stats.mean:.9f}", f"{stats.std:.9f}", stats.count] if stats else ["", "", 0])
                    )
    opponent_path = data_dir / f"{group}_by_opponent.csv"
    with opponent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "opponent_pair", "series", "mean", "std", "count"])
        for metric in METRICS:
            for pair in PAIRS:
                for series in SERIES:
                    stats = opponent_stats(cells, group, metric.key, pair, series.name)
                    writer.writerow(
                        [metric.key, pair, series.name]
                        + ([f"{stats.mean:.9f}", f"{stats.std:.9f}", stats.count] if stats else ["", "", 0])
                    )


def axis_limit(metric: str, maximum: float) -> float:
    fixed = {
        "my_util": 1.0,
        "opp_util1": 1.0,
        "opp_util2": 1.0,
        "social": 3.0,
        "nash": 1.0,
        "step": 80.0,
        "step_efficiency": 0.35,
        "agreement_rate": 100.0,
    }
    return fixed[metric]


def plot_metric(axis, cells, group: str, metric_key: str, domains: tuple[str, ...], title: str) -> None:
    from matplotlib.ticker import PercentFormatter

    series_list = available_series(group)
    plot_domains = (*domains, "Average")
    y_positions = list(range(len(plot_domains)))
    total_height = 0.72
    bar_height = total_height / len(series_list)
    offsets = [(index - (len(series_list) - 1) / 2) * bar_height for index in range(len(series_list))]
    maximum = 0.0
    for offset, series in zip(offsets, series_list):
        means: list[float] = []
        stds: list[float] = []
        for domain in plot_domains:
            stats = domain_stats(cells, group, metric_key, domain, series.name)
            means.append(stats.mean if stats else math.nan)
            stds.append(stats.std if stats else 0.0)
            if stats:
                maximum = max(maximum, stats.mean + stats.std)
        axis.barh(
            [position + offset for position in y_positions], means, height=bar_height,
            color=series.color, label=series.name, xerr=stds,
            error_kw={"ecolor": "0.65", "elinewidth": 0.7, "capsize": 0},
        )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(plot_domains)
    if not axis.yaxis_inverted():
        axis.invert_yaxis()
    axis.set_xlim(0, axis_limit(metric_key, maximum))
    axis.set_xlabel(title)
    axis.grid(False)
    if metric_key == "agreement_rate":
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))


def save_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def table_section_height(metric_keys: tuple[str, ...], header_rows: int) -> float:
    return 1.0 + header_rows * 1.05 + len(metric_keys) * (len(SERIES) * 0.78 + 0.38) + 0.35


def draw_table_section(
    axis,
    cells: dict[CellKey, Accumulator],
    group: str,
    dimension: str,
    items: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> None:
    axis.set_axis_off()
    header_rows = 3 if dimension == "domain" else 1
    height = table_section_height(metric_keys, header_rows)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(height, 0.0)

    label_width = 0.18
    method_width = 0.28
    value_width = (1.0 - label_width - method_width) / len(items)
    label_x = 0.008
    method_x = label_width + 0.012
    value_centers = [label_width + method_width + value_width * (index + 0.5) for index in range(len(items))]
    row_height = 0.78
    y = 0.35

    axis.hlines(y, 0.0, 1.0, color="black", linewidth=1.4)
    y += 0.58

    if dimension == "domain":
        header_data = (
            ("Business Area", [DOMAIN_META[item][0] for item in items]),
            ("Domain", list(items)),
            ("|Ω|", [DOMAIN_META[item][1] for item in items]),
        )
        for label, values in header_data:
            axis.text(label_x, y, label, ha="left", va="center", fontsize=12.5)
            for center, value in zip(value_centers, values):
                axis.text(center, y, value, ha="center", va="center", fontsize=12.5)
            y += 1.05
    else:
        axis.text(label_x, y + 0.18, "Opponent Pair", ha="left", va="center", fontsize=12.5)
        for center, value in zip(value_centers, items):
            left, right = value.split("-", 1)
            axis.text(center, y + 0.18, f"{left}\n{right}", ha="center", va="center", fontsize=12.0, linespacing=0.9)
        y += 1.05

    axis.hlines(y - 0.2, 0.0, 1.0, color="black", linewidth=0.9)
    y += 0.22

    for metric_index, metric_key in enumerate(metric_keys):
        metric = METRIC_BY_KEY[metric_key]
        for series_index, series in enumerate(SERIES):
            axis.text(label_x, y, metric.label if series_index == 0 else "", ha="left", va="center", fontsize=11.7)
            axis.text(method_x, y, series.name, ha="left", va="center", fontsize=11.3)
            for center, item in zip(value_centers, items):
                stats = (
                    domain_stats(cells, group, metric_key, item, series.name)
                    if dimension == "domain"
                    else opponent_stats(cells, group, metric_key, item, series.name)
                )
                axis.text(center, y, format_value(metric, stats), ha="center", va="center", fontsize=11.3)
            y += row_height
        if metric_index < len(metric_keys) - 1:
            axis.hlines(y - 0.2, 0.0, 1.0, color="black", linewidth=0.9)
            y += 0.38

    axis.hlines(y - 0.25, 0.0, 1.0, color="black", linewidth=1.4)


def write_table_images(cells: dict[CellKey, Accumulator], group: str, domains: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "serif"})
    output_root = OUTPUT / "table_images" / group
    domain_sections = (domains[:4], domains[4:]) if group == "known" else (domains,)
    opponent_sections = (PAIRS[:5], PAIRS[5:])

    for filename, metric_keys in TABLE_METRIC_GROUPS:
        for dimension, sections in (("by_domain", domain_sections), ("by_opponent", opponent_sections)):
            header_rows = 3 if dimension == "by_domain" else 1
            section_height = table_section_height(metric_keys, header_rows)
            figure_height = 0.72 * section_height * len(sections) + 0.4 * max(0, len(sections) - 1)
            fig, axes = plt.subplots(len(sections), 1, figsize=(16, figure_height), squeeze=False)
            for axis, items in zip(axes.flat, sections):
                draw_table_section(
                    axis, cells, group,
                    "domain" if dimension == "by_domain" else "opponent",
                    tuple(items), metric_keys,
                )
            fig.subplots_adjust(left=0.025, right=0.99, top=0.99, bottom=0.02, hspace=0.08)
            save_figure(fig, output_root / dimension / filename)
            plt.close(fig)


def write_charts(cells: dict[CellKey, Accumulator], group: str, domains: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    chart_dir = OUTPUT / "charts" / group

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    labels = {
        "my_util": "(a) Utility",
        "step": "(b) Number of steps",
        "step_efficiency": "(c) Step efficiency",
        "agreement_rate": "(d) Agreement rate",
    }
    for axis, metric_key in zip(axes.flat, OVERVIEW_METRICS):
        plot_metric(axis, cells, group, metric_key, domains, labels[metric_key])
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.14, hspace=0.18, wspace=0.30)
    save_figure(fig, chart_dir / f"{group}_domain_metrics_overview")
    plt.close(fig)

    by_metric = chart_dir / "by_metric"
    for metric in METRICS:
        fig, axis = plt.subplots(figsize=(11, max(6.0, 0.9 * (len(domains) + 1))))
        plot_metric(axis, cells, group, metric.key, domains, metric.label)
        handles, legend_labels = axis.get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.18)
        save_figure(fig, by_metric / metric.filename)
        plt.close(fig)


def write_readme() -> None:
    content = """# Negotiation Result Comparison

## Aggregation

- Series: MiPN, RLBOA, Transformer, and alpha-Nego; expert/general are separate.
- MiPN/RLBOA/Transformer: pooled mean over case1-case6 (100 episodes per case).
- alpha-Nego: mean over case1 (100 episodes).
- Known domains: all eight series over seven training domains.
- Unknown domains: general models only. Coffee and SmartPhone are unavailable for alpha-Nego because its checkpoint supports at most five values per issue; these cells are N/A.
- Unknown Average and opponent-pair tables use the fair common subset Camera/Lunch/Kitchen for every general series.
- Agreement rate: percentage of rows where `my_util != 0`, matching the existing summary scripts.
- Step efficiency: per-episode `my_util / step` (zero when step is zero), then averaged.
- Error bars: population standard deviation over all pooled episodes and opponent pairs.

## Outputs

- `charts/{known,unknown}/`: reference-style four-panel charts in PNG/PDF/SVG.
- `charts/{known,unknown}/by_metric/`: one domain bar chart per metric in PNG/PDF/SVG.
- `table_images/{known,unknown}/{by_domain,by_opponent}/`: paper-style overview tables in PNG/PDF/SVG, with two metrics per sheet.
- `tables/{known,unknown}/domain_tables.txt`: all metrics grouped by domain.
- `tables/{known,unknown}/opponent_tables.txt`: all metrics grouped by opponent pair.
- `tables/.../by_domain/` and `by_opponent/`: one text table per metric.
- `data/`: machine-readable means, population standard deviations, and sample counts.

Regenerate with `python generate_comparison.py` from this directory.
"""
    (OUTPUT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    cells = load_cells()
    for group, domains in (("known", KNOWN_DOMAINS), ("unknown", UNKNOWN_DOMAINS)):
        write_summary_csvs(cells, group, domains)
        write_domain_tables(cells, group, domains)
        write_opponent_tables(cells, group)
        write_charts(cells, group, domains)
        write_table_images(cells, group, domains)
    write_readme()
    print("Generated comparison charts, tables, and CSV summaries in", OUTPUT)


if __name__ == "__main__":
    main()
