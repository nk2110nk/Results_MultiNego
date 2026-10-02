# Negotiation Result Comparison

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
