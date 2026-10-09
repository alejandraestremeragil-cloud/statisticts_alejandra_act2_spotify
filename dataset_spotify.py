# -*- coding: utf-8 -*-
"""
Spotify Tracks Dataset - Descriptive statistics and exploratory data analysis (EDA)
Applied Statistics (UAX), Activity 2.

Run it from this folder:   python eda_spotify.py

Input : dataset.csv  (Spotify Tracks Dataset, downloaded from Kaggle)
Output: figures/  -> the 7 figures of the report
        tables/   -> 3 Excel files with the full metrics and the images of Tables 2, 3 and 4
"""

from pathlib import Path

import matplotlib

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
SHOW_PLOTS = False                # True = open every figure in a window; False = only save them
SHOW_KURTOSIS_IN_TABLE = False    # True = add the kurtosis column to the image of Table 2

if not SHOW_PLOTS:
    matplotlib.use("Agg")         # no windows: works on any computer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FIG_DIR = BASE / "figures"
TABLE_DIR = BASE / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

CONTINUOUS = ["popularity", "duration_ms", "danceability", "energy", "loudness",
              "speechiness", "acousticness", "instrumentalness", "liveness",
              "valence", "tempo"]
DISCRETE = ["explicit", "key", "mode", "time_signature"]    # track_genre is analysed separately
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
GREEN = "#1DB954"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def save_figure(name):
    """Save the active figure in figures/ and close it."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def autofit_columns(path):
    """Adjust the column widths of an Excel file."""
    wb = load_workbook(path)
    for ws in wb.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = width + 3
    wb.save(path)


def draw_table(header, rows, widths, path, row_height=0.34):
    """Draw a clean table as an image (used for Tables 2, 3 and 4 of the report)."""
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]}):
        fig, ax = plt.subplots(figsize=(sum(widths), row_height * (len(rows) + 1) + 0.1))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=header,
                         colWidths=[w / sum(widths) for w in widths],
                         loc="center", cellLoc="right", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#999999")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#1F3864")
                cell.set_text_props(color="white", weight="bold", ha="center")
            else:
                cell.set_facecolor("#F2F2F2" if r % 2 else "white")
                if c == 0:
                    cell.set_text_props(weight="bold", ha="left")
                    cell._loc = "left"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)


def category_label(variable, value):
    """Readable label of a category (note names for key, major/minor for mode)."""
    if variable == "key":
        return f"{value} ({NOTES[value]})"
    if variable == "mode":
        return f"{value} ({'major' if value == 1 else 'minor'})"
    return str(value)


# ----------------------------------------------------------------------
# 1. Load and first inspection
# ----------------------------------------------------------------------
df = pd.read_csv(BASE / "dataset.csv", index_col=0)

print("=== 1. FIRST INSPECTION ===")
print("Shape:", df.shape)
nulls = df.isnull().sum()
print("\nColumns with missing values:")
print(nulls[nulls > 0])
print("\nExact duplicate rows:", df.duplicated().sum())
print("Repeated track_id:", df.duplicated(subset="track_id").sum())
print("duration_ms = 0:", (df["duration_ms"] == 0).sum())
print("tempo = 0:", (df["tempo"] == 0).sum())
print("time_signature = 0:", (df["time_signature"] == 0).sum())
print("popularity = 0:", (df["popularity"] == 0).sum())


# ----------------------------------------------------------------------
# 2. Data cleaning (aberrant and missing values)
# ----------------------------------------------------------------------
print("\n=== 2. DATA CLEANING ===")
rows_before = len(df)

# 2.1 Invalid row: no name and duration 0
invalid = df[df["track_name"].isnull() & (df["duration_ms"] == 0)].index
df = df.drop(index=invalid)
print("Invalid rows removed:", len(invalid))

# 2.2 Exact duplicates (repeated track_id are kept: the same song appears in several genres)
n_duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print("Exact duplicates removed:", n_duplicates)

# 2.3 Impossible zeros -> missing values
df["tempo"] = df["tempo"].replace(0, np.nan)
df["time_signature"] = df["time_signature"].replace(0, np.nan)
print("Missing tempo:", df["tempo"].isnull().sum(),
      "| Missing time_signature:", df["time_signature"].isnull().sum())

# 2.4 Replacement: median for tempo (continuous) and mode for time_signature (discrete)
df["tempo"] = df["tempo"].fillna(df["tempo"].median())
df["time_signature"] = df["time_signature"].fillna(df["time_signature"].mode()[0]).astype(int)

print(f"Rows: {rows_before} -> {len(df)}")
print(f"Songs with popularity = 0: {(df['popularity'] == 0).mean() * 100:.1f}%")


# ----------------------------------------------------------------------
# 3. Descriptive statistics metrics (Tables 2, 3 and 4)
# ----------------------------------------------------------------------
print("\n=== 3. TABLES ===")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# 3.1 Continuous variables (Table 2)
cont_table = pd.DataFrame({
    "n": df[CONTINUOUS].count(),
    "mean": df[CONTINUOUS].mean(),
    "median": df[CONTINUOUS].median(),
    "std_dev": df[CONTINUOUS].std(),
    "min": df[CONTINUOUS].min(),
    "Q1": df[CONTINUOUS].quantile(0.25),
    "Q3": df[CONTINUOUS].quantile(0.75),
    "max": df[CONTINUOUS].max(),
    "asymmetry": df[CONTINUOUS].skew(),
    "kurtosis": df[CONTINUOUS].kurt(),      # excess kurtosis (a normal distribution = 0)
}).round(3)
print(cont_table)
cont_table.to_excel(TABLE_DIR / "continuous_variables.xlsx")

cols = ["mean", "median", "std_dev", "min", "Q1", "Q3", "max", "asymmetry"]
header = ["Variable", "Mean", "Median", "Std. dev.", "Min", "Q1", "Q3", "Max", "Asymmetry"]
widths = [1.9, 1.15, 1.15, 1.1, 1.1, 1.0, 1.0, 1.15, 1.1]
if SHOW_KURTOSIS_IN_TABLE:
    cols.append("kurtosis")
    header.append("Kurtosis")
    widths.append(1.0)
rows = []
for var in cont_table.index:
    values = []
    for c in cols:
        x = cont_table.loc[var, c]
        if var == "duration_ms" and c not in ("asymmetry", "kurtosis"):
            values.append(f"{x:,.0f}")          # milliseconds without decimals
        else:
            values.append(f"{x:,.3f}")
    rows.append([var] + values)
draw_table(header, rows, widths, TABLE_DIR / "table2_continuous_metrics.png")

# 3.2 Discrete variables (Table 3)
with pd.ExcelWriter(TABLE_DIR / "discrete_variables.xlsx") as writer:
    for col in DISCRETE:
        t = df[col].value_counts().sort_index().to_frame("frequency")
        t["percentage"] = (t["frequency"] / t["frequency"].sum() * 100).round(2)
        print(f"\n{col}\n{t}\nMode: {df[col].mode()[0]}")
        t.to_excel(writer, sheet_name=col)

rows = []
for col in DISCRETE:
    counts = df[col].value_counts()                 # sorted from most to least frequent
    rows.append([col, str(len(counts)),
                 category_label(col, counts.index[0]), f"{counts.iloc[0]:,}",
                 f"{counts.iloc[0] / len(df) * 100:.2f}%",
                 category_label(col, counts.index[-1]),
                 f"{counts.iloc[-1] / len(df) * 100:.2f}%"])
draw_table(["Variable", "Categories", "Most frequent", "Frequency", "% of songs",
            "Least frequent", "% of songs"],
           rows, [1.9, 1.2, 1.5, 1.3, 1.3, 1.6, 1.3],
           TABLE_DIR / "table3_discrete_metrics.png", row_height=0.42)

print("\nDifferent genres:", df["track_genre"].nunique())
print(df["track_genre"].value_counts().describe().round(1))

# 3.3 Outliers (boxplot criterion: 1.5 x IQR) (Table 4)
records = []
for col in CONTINUOUS:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    below = int((df[col] < lower).sum())
    above = int((df[col] > upper).sum())
    records.append({"variable": col,
                    "lower limit": round(lower, 3), "upper limit": round(upper, 3),
                    "outliers below": below, "outliers above": above,
                    "total": below + above,
                    "% of songs": round((below + above) / len(df) * 100, 1)})
outliers_table = pd.DataFrame(records).set_index("variable")
print("\n", outliers_table)
outliers_table.to_excel(TABLE_DIR / "outliers.xlsx")

ordered = outliers_table.reset_index().sort_values(["total", "variable"])
rows = [[r["variable"], f"{int(r['total']):,}", f"{r['% of songs']:.1f}%"]
        for _, r in ordered.iterrows()]
draw_table(["Variable", "Outliers", "% of songs"], rows, [1.9, 1.2, 1.2],
           TABLE_DIR / "table4_outliers.png")

for name in ["continuous_variables", "discrete_variables", "outliers"]:
    autofit_columns(TABLE_DIR / f"{name}.xlsx")


# ----------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------
print("\n=== 4. FIGURES ===")

# Figure 1: histograms with mean and median
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, col in zip(axes.flat, CONTINUOUS):
    data = df[col].dropna()
    mean, median = data.mean(), data.median()

    title = col
    if col == "duration_ms":    # a few very long songs flatten the plot: limit to the 99th percentile
        data = data[data <= data.quantile(0.99)]
        title = "duration_ms (up to 99th percentile)"

    ax.hist(data, bins=40, color=GREEN, edgecolor="white")
    ax.axvline(median, color="black", linestyle="-", linewidth=1.8, label=f"Median: {median:,.2f}")
    ax.axvline(mean, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {mean:,.2f}")
    ax.set_title(title)
    ax.set_ylabel("Number of songs")
    ax.legend(fontsize=8)
axes.flat[-1].axis("off")
save_figure("fig1_histograms.png")

# Figure 2: bar charts of the discrete variables (percentages, most frequent category in green)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.flat, DISCRETE):
    pct = df[col].value_counts(normalize=True).sort_index() * 100

    if col == "key":
        labels = [f"{i}\n{NOTES[i]}" for i in pct.index]
    else:
        labels = [str(i) for i in pct.index]

    colors = [GREEN if v == pct.max() else "#B3B3B3" for v in pct.values]
    bars = ax.bar(labels, pct.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in pct.values], padding=2, fontsize=9)
    ax.set_title(f"{col} (most frequent in green)")
    ax.set_ylabel("% of songs")
    ax.set_ylim(0, pct.max() * 1.15)
save_figure("fig2_discrete_bars.png")

# Figure 3: boxplots
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, CONTINUOUS):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
axes.flat[-1].axis("off")
save_figure("fig3_boxplots.png")

# Figure 4: correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[CONTINUOUS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlations")
save_figure("fig4_correlations.png")

# Figure 5: scatter plot energy vs loudness (random sample; the correlation uses all the songs)
sample = df.sample(min(8000, len(df)), random_state=1)
plt.figure(figsize=(7, 5))
sns.regplot(data=sample, x="loudness", y="energy",
            scatter_kws={"alpha": 0.15, "s": 8, "color": GREEN},
            line_kws={"color": "red"})
plt.title(f"Energy vs Loudness (sample of {len(sample):,} songs)")
plt.ylim(-0.05, 1.05)
save_figure("fig5_scatter_energy_loudness.png")

# Figure 6: top 10 genres by average popularity
genre_means = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False)
print("\nTop 10 genres by average popularity:")
print(genre_means.head(10).round(2))
print("\n5 least popular genres:")
print(genre_means.tail(5).round(2))
print(f"\nAverage of the genre means: {genre_means.mean():.2f} (std. dev. {genre_means.std():.2f})")

plt.figure(figsize=(8, 5))
genre_means.head(10).sort_values().plot(kind="barh", title="Top 10 Genres by Average Popularity")
plt.xlabel("Average popularity")
save_figure("fig6_top_genres.png")


# ----------------------------------------------------------------------
# 5. Feature engineering
# ----------------------------------------------------------------------
print("\n=== 5. FEATURE ENGINEERING ===")

# Derived variables
df["duration_min"] = df["duration_ms"] / 60000            # duration in minutes
df["log_duration"] = np.log(df["duration_ms"])            # reduces the asymmetry (duration > 0 after cleaning)
df["explicit_int"] = df["explicit"].astype(int)           # dummy variable 0/1

# Numerical -> categorical
df["popularity_level"] = pd.cut(df["popularity"], bins=[-1, 20, 50, 100],
                                labels=["low", "medium", "high"])
print("\nPopularity level (%):")
print((df["popularity_level"].value_counts(normalize=True) * 100).round(1))

# Dummy variables (n categories -> n-1 columns)
dummies = pd.get_dummies(df["time_signature"], prefix="time_sig", drop_first=True).astype(int)
df = pd.concat([df, dummies], axis=1)
print("\nDummy columns created:", list(dummies.columns))

# Effect of the logarithm on the asymmetry of the duration
asym_original = df["duration_ms"].skew()
asym_log = df["log_duration"].skew()
print(f"\nAsymmetry of duration_ms: {asym_original:.2f} -> log(duration_ms): {asym_log:.2f}")
print(f"Kurtosis of duration_ms: {df['duration_ms'].kurt():.2f} -> log(duration_ms): {df['log_duration'].kurt():.2f}")

# Figure 7: duration before and after the logarithm
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
cut = df["duration_ms"].quantile(0.99)
axes[0].hist(df.loc[df["duration_ms"] <= cut, "duration_ms"], bins=40, color=GREEN, edgecolor="white")
axes[0].set_title(f"duration_ms (up to 99th percentile)\nasymmetry coefficient = {asym_original:.2f} (all songs)")
axes[0].set_xlabel("Duration (ms)")
axes[0].set_ylabel("Number of songs")
axes[1].hist(df["log_duration"], bins=40, color=GREEN, edgecolor="white")
axes[1].set_title(f"log(duration_ms)\nasymmetry coefficient = {asym_log:.2f}")
axes[1].set_xlabel("log(duration in ms)")
axes[1].set_ylabel("Number of songs")
save_figure("fig7_log_duration.png")


# ----------------------------------------------------------------------
print("\nDone. Check the folders 'figures' and 'tables'.")
