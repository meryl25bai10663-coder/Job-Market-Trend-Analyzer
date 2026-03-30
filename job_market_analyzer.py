#   Dataset : LinkedIn Job Postings (2023–2024)
#   Source  : https://www.kaggle.com/datasets/arshkon/linkedin-job-postings


import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (13, 6)

# ─── sklearn imports (needed only for Phase 5) ───────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ════════════════════════════════════════════════════════════
#   SHARED HELPERS
# ════════════════════════════════════════════════════════════

_state = {
    "postings"      : None,
    "skills_df"     : None,
    "industries_df" : None,
    "salaries_df"   : None,
    "best_model"    : None,
    "best_name"     : None,
    "feature_cols"  : None,
    "le_dict"       : None,
    "work_col"      : None,
    "ind_name_col"  : None,
    "HAS_INDUSTRY"  : False,
}

SKILL_NAMES = {
    "MRKT": "Marketing",         "PR":   "Public Relations",
    "WRT":  "Writing",           "SALE": "Sales",
    "FIN":  "Finance",           "ADVR": "Advertising",
    "BD":   "Business Dev.",     "ENG":  "Engineering",
    "PRJM": "Project Mgmt",      "IT":   "Information Technology",
    "GENB": "General Business",  "ADM":  "Administration",
    "ACCT": "Accounting",        "CUST": "Customer Service",
    "HR":   "Human Resources",   "LEGL": "Legal",
    "HLTH": "Healthcare",        "RSCH": "Research",
    "DIST": "Distribution",      "MNFG": "Manufacturing",
    "SUPL": "Supply Chain",      "EDUC": "Education",
    "DSGN": "Design",            "DATA": "Data Analysis",
    "SOFT": "Software Dev.",     "MGMT": "Management",
    "OPS":  "Operations",        "PROD": "Product Mgmt",
    "ANLZ": "Analytics",         "COMM": "Communication",
    "MACH": "Machine Learning",  "CLUD": "Cloud Computing",
    "CYBR": "Cybersecurity",     "ARTD": "Artificial Intelligence",
    "STAT": "Statistics",        "VISL": "Data Visualization",
    "PRGM": "Programming",       "NTWK": "Networking",
    "DBAS": "Database Mgmt",     "MOBL": "Mobile Dev.",
}

CATEGORIES = {
    "Programming & Tech" : ["ENG", "IT", "SOFT", "PRGM", "MOBL", "NTWK", "DBAS"],
    "Data & Analytics"   : ["DATA", "ANLZ", "STAT", "VISL", "RSCH", "MACH"],
    "Business"           : ["SALE", "BD", "GENB", "MGMT", "OPS", "PROD", "ACCT", "FIN"],
    "Marketing & Comms"  : ["MRKT", "ADVR", "PR", "WRT", "COMM", "CUST"],
    "AI & Cloud"         : ["ARTD", "MACH", "CLUD", "CYBR", "STAT"],
    "Other"              : ["ADM", "HR", "LEGL", "HLTH", "EDUC", "DIST", "MNFG", "SUPL", "PRJM"],
}


def _header(title: str):
    width = 57
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title: str):
    print(f"\n  -- {title} --")
    print("  " + "-" * (len(title) + 6))


def _print_table(df: pd.DataFrame, title: str = ""):
    """Print a DataFrame as a clean ranked text table."""
    if title:
        _subheader(title)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    print(df.to_string())
    print()


def _check_file(filename: str) -> bool:
    if os.path.isfile(filename):
        return True
    print(f"\n  [X]  '{filename}' not found.")
    print(f"       Place it in: {os.path.abspath('.')}")
    return False


def _to_yearly(row):
    multiplier = 2080 if str(row.get("pay_period", "")).upper() == "HOURLY" else 1
    for col in ["med_salary", "max_salary", "min_salary"]:
        if pd.notna(row.get(col)):
            return row[col] * multiplier
    if pd.notna(row.get("max_salary")) and pd.notna(row.get("min_salary")):
        return ((row["max_salary"] + row["min_salary"]) / 2) * multiplier
    return None


def _load_cleaned_postings():
    if _state["postings"] is not None:
        return _state["postings"]
    src = "cleaned_job_postings.csv"
    if os.path.isfile(src):
        _state["postings"] = pd.read_csv(src)
        print(f"  [OK] Loaded cleaned data ({_state['postings'].shape[0]:,} rows).")
        return _state["postings"]
    print("  [..] Running Phase 1 to generate cleaned data first ...")
    run_phase1(silent=True)
    return _state["postings"]


def _load_salaries():
    if _state["salaries_df"] is not None:
        return _state["salaries_df"]
    if not _check_file("salaries.csv"):
        return None
    sal = pd.read_csv("salaries.csv")
    for col in ["max_salary", "med_salary", "min_salary"]:
        sal[col] = pd.to_numeric(sal[col], errors="coerce")
    sal["salary"] = sal.apply(_to_yearly, axis=1)
    if "compensation_type" in sal.columns:
        sal = sal[sal["compensation_type"] == "BASE_SALARY"]
    sal = sal[(sal["salary"] >= 20_000) & (sal["salary"] <= 500_000)]
    _state["salaries_df"] = sal
    return sal


def _load_industries():
    if _state["industries_df"] is not None:
        ind = _state["industries_df"]
        id_col   = "job_id" if "job_id" in ind.columns else ind.columns[0]
        name_col = next(
            (c for c in ["industry_name", "industry", "name", "sector"] if c in ind.columns),
            ind.columns[-1]
        )
        return ind, id_col, name_col
    if not _check_file("job_industries.csv"):
        return None, None, None
    ind = pd.read_csv("job_industries.csv")
    ind.columns = [c.strip().lower() for c in ind.columns]
    id_col   = "job_id" if "job_id" in ind.columns else ind.columns[0]
    name_col = next(
        (c for c in ["industry_name", "industry", "name", "sector"] if c in ind.columns),
        ind.columns[-1]
    )
    _state["industries_df"] = ind
    return ind, id_col, name_col


def run_phase1(silent: bool = False):
    if not silent:
        _header("PHASE 1 -- Data Loading, Cleaning & EDA")

    if not _check_file("job_postings.csv"):
        return

    df = pd.read_csv("job_postings.csv")
    if not silent:
        print(f"\n  Rows    : {df.shape[0]:,}")
        print(f"  Columns : {df.shape[1]}")

    # -- Missing values -> TEXT TABLE ----------------------------
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({
        "Missing Count": missing,
        "Missing %"    : missing_pct
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

    if not silent:
        _print_table(missing_df, "Missing Values Summary")

    # -- Clean ---------------------------------------------------
    cols_to_drop = missing_pct[missing_pct > 70].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    before = len(df)
    df.drop_duplicates(inplace=True)

    if not silent:
        print(f"  Dropped {len(cols_to_drop)} high-missing columns")
        print(f"  Removed {before - len(df)} duplicate rows")

    for col in ["title", "company_name", "location", "work_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    for col in ["max_salary", "min_salary", "med_salary"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "original_listed_time" in df.columns:
        df["posted_date"]  = pd.to_datetime(df["original_listed_time"], unit="ms", errors="coerce")
        df["posted_month"] = df["posted_date"].dt.to_period("M")

    df.to_csv("cleaned_job_postings.csv", index=False)
    _state["postings"] = df

    if not silent:
        print(f"\n  [OK] Cleaned data saved -> cleaned_job_postings.csv ({df.shape[0]:,} rows)")

    if silent:
        return

    # -- Top job titles -> TEXT TABLE ----------------------------
    if "title" in df.columns:
        top_titles = df["title"].value_counts().head(15).reset_index()
        top_titles.columns = ["Job Title", "Postings"]
        _print_table(top_titles, "Top 15 Most Common Job Titles")

    # -- Top locations -> TEXT TABLE -----------------------------
    if "location" in df.columns:
        top_loc = df["location"].value_counts().head(15).reset_index()
        top_loc.columns = ["Location", "Postings"]
        _print_table(top_loc, "Top 15 Job Posting Locations")

    # -- Work type -> CHART (proportions best shown visually) ----
    if "work_type" in df.columns:  
        wc = df["work_type"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 7))
        wedges, texts, autotexts = ax.pie(wc.values,labels=None,autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"), pctdistance=0.85, wedgeprops={"linewidth": 1, "edgecolor": "white"})
        # Move percentage labels outward for small slices
        for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            angle = (wedge.theta1 + wedge.theta2) / 2
            x = 1.1 * np.cos(np.radians(angle))
            y = 1.1 * np.sin(np.radians(angle))
            autotext.set_position((x, y))
            autotext.set_fontsize(9)
        ax.legend(wedges, [f"{label} ({val:,})" for label, val in zip(wc.index, wc.values)], title="Work Type", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        ax.set_title("Work Type Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("work_type_distribution.png", dpi=150, bbox_inches="tight")
        plt.show()
        print('  [OK] Chart saved: work_type_distribution.png')

    # -- Postings over time -> CHART (trend line) ----------------
    if "posted_month" in df.columns:
        pot = df["posted_month"].value_counts().sort_index()
        plt.figure(figsize=(13, 5))
        pot.plot(kind="line", marker="o", color="steelblue", linewidth=2)
        plt.title("Job Postings Over Time (Monthly)", fontsize=14, fontweight="bold")
        plt.xlabel("Month"); plt.ylabel("Number of Postings")
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig("postings_over_time.png", dpi=150)
        plt.show()
        print("  [OK] Chart saved: postings_over_time.png")

    # -- Experience level -> TEXT SUMMARY ------------------------
    if "formatted_experience_level" in df.columns:
        exp = df["formatted_experience_level"].value_counts().reset_index()
        exp.columns = ["Experience Level", "Postings"]
        exp["Share %"] = (exp["Postings"] / exp["Postings"].sum() * 100).round(1)
        _print_table(exp, "Job Postings by Experience Level")

    print("  Phase 1 complete!")


#   PHASE 2 — Skills Analysis


def run_phase2():
    from collections import Counter

    _header("PHASE 2 -- Skills Analysis")

    if not _check_file("job_skills.csv"):
        return

    postings  = _load_cleaned_postings()
    skills_df = pd.read_csv("job_skills.csv")
    skills_df.columns = [c.strip().lower() for c in skills_df.columns]
    skills_df["skill_name"] = skills_df["skill_abr"].map(SKILL_NAMES).fillna(skills_df["skill_abr"])
    _state["skills_df"] = skills_df

    print(f"  [OK] Skills loaded: {skills_df.shape[0]:,} rows")

    # -- Top 20 skills -> TEXT TABLE -----------------------------
    skill_counts = Counter(skills_df["skill_name"].tolist())
    top20 = pd.DataFrame(skill_counts.most_common(20), columns=["Skill", "Job Postings"])
    _print_table(top20, "Top 20 Most In-Demand Skills")

    # -- Skill categories -> TEXT SUMMARY ------------------------
    abr_counts = Counter(skills_df["skill_abr"].tolist())
    category_counts = {
        cat: sum(abr_counts.get(code, 0) for code in codes)
        for cat, codes in CATEGORIES.items()
    }
    cat_df = pd.DataFrame(
        list(category_counts.items()), columns=["Category", "Total Mentions"]
    ).sort_values("Total Mentions", ascending=False)

    _subheader("Skill Categories in Demand")
    total_mentions = cat_df["Total Mentions"].sum()
    for _, row in cat_df.iterrows():
        bar = "#" * int(row["Total Mentions"] / total_mentions * 40)
        pct = row["Total Mentions"] / total_mentions * 100
        print(f"  {row['Category']:<22} {row['Total Mentions']:>7,}  ({pct:4.1f}%)  {bar}")

    # -- Skills by experience level -> TEXT TABLE ----------------
    if "job_id" in skills_df.columns and "formatted_experience_level" in postings.columns:
        id_col = "job_id" if "job_id" in postings.columns else postings.columns[0]
        merged = skills_df.merge(
            postings[[id_col, "formatted_experience_level"]],
            left_on="job_id", right_on=id_col, how="left"
        )
        exp_levels = merged["formatted_experience_level"].dropna().value_counts().head(4).index.tolist()
        _subheader("Top 10 Skills by Experience Level")
        for level in exp_levels:
            subset = merged[merged["formatted_experience_level"] == level]
            ts = subset["skill_name"].value_counts().head(10).reset_index()
            ts.columns = ["Skill", "Count"]
            print(f"\n  [ {level} ]")
            print(ts.to_string(index=False))

    top20.to_csv("top20_skills_summary.csv", index=False)
    print("\n  [OK] Saved: top20_skills_summary.csv")
    print("\n  Phase 2 complete!")


#   PHASE 3 — Sector-wise Trend Analysis


def run_phase3():
    _header("PHASE 3 -- Sector-wise Trend Analysis")

    postings = _load_cleaned_postings()
    ind_df, id_col_ind, ind_name_col = _load_industries()

    if ind_df is not None:
        id_col_post = "job_id" if "job_id" in postings.columns else postings.columns[0]
        merged = postings.merge(
            ind_df[[id_col_ind, ind_name_col]],
            left_on=id_col_post, right_on=id_col_ind, how="left"
        )
        merged[ind_name_col] = merged[ind_name_col].astype(str).str.strip().str.title()
        print(f"  [OK] Merged: {merged.shape[0]:,} rows | {merged[ind_name_col].nunique()} sectors")
    else:
        fallback = next((c for c in ["industry", "sector", "job_type"] if c in postings.columns), None)
        if fallback:
            merged = postings.copy()
            ind_name_col = fallback
        else:
            print("  [X]  No industry data available. Please add job_industries.csv.")
            return

    # -- Top 15 sectors -> TEXT TABLE ----------------------------
    top_sectors = merged[ind_name_col].value_counts().head(15).reset_index()
    top_sectors.columns = ["Sector", "Job Postings"]
    _print_table(top_sectors, "Top 15 Hiring Sectors")

    # -- Sector share -> TEXT BAR CHART --------------------------
    _subheader("Sector Share of Job Postings (Top 10)")
    total = len(merged)
    top10_counts = merged[ind_name_col].value_counts().head(10)
    for sector, count in top10_counts.items():
        bar = "#" * int(count / total * 50)
        pct = count / total * 100
        print(f"  {sector:<35} {pct:5.1f}%  {bar}")
    other_pct = merged[ind_name_col].value_counts().iloc[10:].sum() / total * 100
    print(f"  {'Other':<35} {other_pct:5.1f}%")

    # -- Sector trends over time -> CHART (multi-line trend) -----
    date_col = next((c for c in ["posted_month", "posted_date"] if c in merged.columns), None)
    if date_col:
        top5 = merged[ind_name_col].value_counts().head(5).index.tolist()
        trend_df = merged[merged[ind_name_col].isin(top5)]
        trend_grouped = (
            trend_df.groupby([date_col, ind_name_col])
            .size().reset_index(name="count")
        )
        plt.figure(figsize=(14, 6))
        for sector in top5:
            s = trend_grouped[trend_grouped[ind_name_col] == sector]
            plt.plot(s[date_col].astype(str), s["count"],
                     marker="o", linewidth=2, label=sector)
        plt.title("Job Posting Trends Over Time -- Top 5 Sectors",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Month"); plt.ylabel("Number of Postings")
        plt.xticks(rotation=45)
        plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("sector_trends_over_time.png", dpi=150)
        plt.show()
        print("\n  [OK] Chart saved: sector_trends_over_time.png")

    # -- Sector by work type -> PIVOT TEXT TABLE -----------------
    if "work_type" in merged.columns:
        top8 = merged[ind_name_col].value_counts().head(8).index.tolist()
        pivot = (
            merged[merged[ind_name_col].isin(top8)]
            .groupby([ind_name_col, "work_type"])
            .size().unstack(fill_value=0)
        )
        _subheader("Work Type Distribution Across Top 8 Sectors")
        print(pivot.to_string())

    # -- Sector by experience level -> PIVOT TEXT TABLE ----------
    if "formatted_experience_level" in merged.columns:
        top6 = merged[ind_name_col].value_counts().head(6).index.tolist()
        pivot_exp = (
            merged[merged[ind_name_col].isin(top6)]
            .groupby([ind_name_col, "formatted_experience_level"])
            .size().unstack(fill_value=0)
        )
        _subheader("Experience Level Distribution Across Top 6 Sectors")
        print(pivot_exp.to_string())

    merged[ind_name_col].value_counts().reset_index().rename(
        columns={ind_name_col: "Sector", "count": "Job_Postings"}
    ).to_csv("sector_summary.csv", index=False)
    print("\n  [OK] Saved: sector_summary.csv")
    print("\n  Phase 3 complete!")


#   PHASE 4 — Salary Analysis


def run_phase4():
    _header("PHASE 4 -- Salary Analysis")

    postings = _load_cleaned_postings()
    sal      = _load_salaries()
    if sal is None:
        return

    ind_df, id_col_ind, ind_name_col = _load_industries()
    HAS_INDUSTRY = ind_df is not None

    id_col = "job_id" if "job_id" in postings.columns else postings.columns[0]
    df     = postings.merge(sal[["job_id", "salary"]], left_on=id_col,
                            right_on="job_id", how="inner")

    if HAS_INDUSTRY:
        df = df.merge(ind_df[[id_col_ind, ind_name_col]],
                      left_on=id_col, right_on=id_col_ind, how="left")
        df[ind_name_col] = df[ind_name_col].astype(str).str.strip().str.title()

    print(f"  [OK] Rows with salary data: {df.shape[0]:,}")

    # -- Overall stats -> TEXT -----------------------------------
    s = df["salary"]
    _subheader("Overall Salary Statistics")
    print(f"  Mean          : ${s.mean():>12,.0f}")
    print(f"  Median        : ${s.median():>12,.0f}")
    print(f"  Std Dev       : ${s.std():>12,.0f}")
    print(f"  Min           : ${s.min():>12,.0f}")
    print(f"  25th pct      : ${s.quantile(0.25):>12,.0f}")
    print(f"  75th pct      : ${s.quantile(0.75):>12,.0f}")
    print(f"  Max           : ${s.max():>12,.0f}")

    # -- Salary distribution -> CHART (shape of data) ------------
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.hist(s.dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Salary Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annual Salary (USD)"); ax.set_ylabel("Number of Jobs")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.suptitle("Overall Salary Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("salary_distribution.png", dpi=150)
    plt.show()
    print("\n  [OK] Chart saved: salary_distribution.png")

    # -- Salary by experience level -> TEXT TABLE ----------------
    if "formatted_experience_level" in df.columns:
        exp_sal = (
            df.dropna(subset=["formatted_experience_level", "salary"])
            .groupby("formatted_experience_level")["salary"]
            .agg(Median="median", Mean="mean", Min="min", Max="max", Count="count")
            .sort_values("Median", ascending=False)
        )
        exp_sal_fmt = exp_sal.copy()
        for col in ["Median", "Mean", "Min", "Max"]:
            exp_sal_fmt[col] = exp_sal_fmt[col].apply(lambda x: f"${x:,.0f}")
        _subheader("Salary by Experience Level")
        print(exp_sal_fmt.to_string())

    # -- Salary by work type -> TEXT TABLE -----------------------
    work_col = next((c for c in ["formatted_work_type", "work_type"] if c in df.columns), None)
    if work_col:
        wt_sal = (
            df.dropna(subset=[work_col, "salary"])
            .groupby(work_col)["salary"]
            .agg(Median="median", Mean="mean", Min="min", Max="max", Count="count")
            .sort_values("Median", ascending=False)
        )
        wt_sal_fmt = wt_sal.copy()
        for col in ["Median", "Mean", "Min", "Max"]:
            wt_sal_fmt[col] = wt_sal_fmt[col].apply(lambda x: f"${x:,.0f}")
        _subheader("Salary by Work Type")
        print(wt_sal_fmt.to_string())

    # -- Top 10 paying sectors -> TEXT TABLE ---------------------
    if HAS_INDUSTRY:
        sector_sal = (
            df.dropna(subset=[ind_name_col, "salary"])
            .groupby(ind_name_col)["salary"]
            .agg(["median", "mean", "count"])
            .query("count >= 10")
            .sort_values("median", ascending=False)
            .head(10)
            .reset_index()
        )
        sector_sal.columns = ["Sector", "Median Salary", "Mean Salary", "Count"]
        sector_sal["Median Salary"] = sector_sal["Median Salary"].apply(lambda x: f"${x:,.0f}")
        sector_sal["Mean Salary"]   = sector_sal["Mean Salary"].apply(lambda x: f"${x:,.0f}")
        _print_table(sector_sal, "Top 10 Highest Paying Sectors")

    # -- Top 15 paying titles -> TEXT TABLE ----------------------
    if "title" in df.columns:
        title_sal = (
            df.dropna(subset=["title", "salary"])
            .groupby("title")["salary"]
            .agg(["median", "count"])
            .query("count >= 5")
            .sort_values("median", ascending=False)
            .head(15)
            .reset_index()
        )
        title_sal.columns = ["Job Title", "Median Salary", "Count"]
        title_sal["Median Salary"] = title_sal["Median Salary"].apply(lambda x: f"${x:,.0f}")
        _print_table(title_sal, "Top 15 Highest Paying Job Titles")

    df[["title", "salary"]].dropna().to_csv("salary_summary.csv", index=False)
    print("  [OK] Saved: salary_summary.csv")
    print("\n  Phase 4 complete!")



#   PHASE 5 — ML Salary Predictor


def run_phase5():
    _header("PHASE 5 -- ML Salary Predictor")

    if not SKLEARN_AVAILABLE:
        print("  [X]  scikit-learn not installed. Run: pip install scikit-learn")
        return

    postings = _load_cleaned_postings()
    sal      = _load_salaries()
    if sal is None:
        return

    ind_df, id_col_ind, ind_name_col = _load_industries()
    HAS_INDUSTRY = ind_df is not None

    id_col = "job_id" if "job_id" in postings.columns else postings.columns[0]
    df     = postings.merge(sal[["job_id", "salary"]], left_on=id_col,
                            right_on="job_id", how="inner")

    if HAS_INDUSTRY:
        df = df.merge(ind_df[[id_col_ind, ind_name_col]],
                      left_on=id_col, right_on=id_col_ind, how="left")
        df[ind_name_col] = df[ind_name_col].astype(str).str.strip().str.title()

    print(f"  [OK] Dataset ready: {df.shape[0]:,} rows")

    # -- Feature engineering -------------------------------------
    work_col     = next((c for c in ["formatted_work_type", "work_type"] if c in df.columns), None)
    feature_cols = []
    if work_col:
        feature_cols.append(work_col)
    if "formatted_experience_level" in df.columns:
        feature_cols.append("formatted_experience_level")
    if HAS_INDUSTRY and ind_name_col:
        feature_cols.append(ind_name_col)
    if "title" in df.columns:
        feature_cols.append("title")

    model_df = df[feature_cols + ["salary"]].dropna()
    print(f"  Features       : {feature_cols}")
    print(f"  Rows (dropna)  : {model_df.shape[0]:,}")

    le_dict = {}
    for col in feature_cols:
        le = LabelEncoder()
        if col == "title":
            top_titles = model_df["title"].value_counts().head(100).index
            model_df["title"] = model_df["title"].where(
                model_df["title"].isin(top_titles), other="Other"
            )
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        le_dict[col]  = le

    X, y = model_df[feature_cols], model_df["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train : {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # -- Train models --------------------------------------------
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name} ...", end="", flush=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {"model": model, "preds": preds, "MAE": mae, "R2": r2}
        print("  done.")

    # -- Model comparison -> TEXT TABLE --------------------------
    best_r2 = max(v["R2"] for v in results.values())
    _subheader("Model Performance Comparison")
    print(f"  {'Model':<25} {'R2 Score':>10}  {'MAE (USD)':>12}  Note")
    print(f"  {'-'*25} {'-'*10}  {'-'*12}  {'-'*15}")
    for name, v in results.items():
        note = "<-- Best Model" if v["R2"] == best_r2 else ""
        print(f"  {name:<25} {v['R2']:>10.4f}  ${v['MAE']:>11,.0f}  {note}")

    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]
    best_preds = results[best_name]["preds"]

    print(f"\n  Best Model  : {best_name}")
    print(f"  R2 Score    : {results[best_name]['R2']:.4f}")
    print(f"  MAE         : ${results[best_name]['MAE']:,.0f}  (avg prediction error)")

    # -- Cache ---------------------------------------------------
    _state.update({
        "best_model"   : best_model,
        "best_name"    : best_name,
        "feature_cols" : feature_cols,
        "le_dict"      : le_dict,
        "work_col"     : work_col,
        "ind_name_col" : ind_name_col,
        "HAS_INDUSTRY" : HAS_INDUSTRY,
    })

    # -- Actual vs Predicted -> CHART (essential for ML eval) ----
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_preds, alpha=0.3, color="steelblue",
                edgecolors="none", s=15)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="red", linewidth=2, linestyle="--", label="Perfect Prediction")
    plt.title(f"Actual vs Predicted Salary -- {best_name}",
              fontsize=14, fontweight="bold")
    plt.xlabel("Actual Salary (USD)"); plt.ylabel("Predicted Salary (USD)")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.legend(); plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("\n  [OK] Chart saved: actual_vs_predicted.png")

    # -- Feature importance -> TEXT TABLE ------------------------
    if hasattr(best_model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature"    : feature_cols,
            "Importance" : best_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        _subheader(f"Feature Importance -- {best_name}")
        print(f"  {'Feature':<30} {'Importance %':>14}  Visual")
        print(f"  {'-'*30} {'-'*14}  {'-'*20}")
        for _, row in imp_df.iterrows():
            bar = "#" * int(row["Importance"] * 50)
            pct = row["Importance"] * 100
            print(f"  {row['Feature']:<30} {pct:>13.2f}%  {bar}")

    pd.DataFrame([
        {"Model": k, "R2_Score": round(v["R2"], 4), "MAE_USD": round(v["MAE"], 2)}
        for k, v in results.items()
    ]).to_csv("model_results.csv", index=False)
    print("\n  [OK] Saved: model_results.csv")
    print("\n  Phase 5 complete!")


# ════════════════════════════════════════════════════════════
#   PHASE 5b — Interactive Salary Predictor
# ════════════════════════════════════════════════════════════

def run_salary_predictor():
    _header("Interactive Salary Predictor")

    if _state["best_model"] is None:
        print("  [!]  Model not trained yet. Running Phase 5 first ...")
        run_phase5()
        if _state["best_model"] is None:
            return

    best_model   = _state["best_model"]
    best_name    = _state["best_name"]
    feature_cols = _state["feature_cols"]
    le_dict      = _state["le_dict"]
    work_col     = _state["work_col"]
    ind_name_col = _state["ind_name_col"]
    HAS_INDUSTRY = _state["HAS_INDUSTRY"]

    print(f"\n  Using model : {best_name}")
    print("  Type 'quit' at any prompt to return to menu.\n")

    def _show_options(col):
        if col in le_dict:
            opts = list(le_dict[col].classes_)
            preview = ", ".join(opts[:8]) + (" ..." if len(opts) > 8 else "")
            print(f"    Options: {preview}")

    def _encode(col, value):
        if value and col in le_dict:
            classes = le_dict[col].classes_
            v = value if value in classes else classes[0]
            return int(le_dict[col].transform([v])[0])
        return 0

    while True:
        print("\n  " + "-" * 50)

        print("  Job Title")
        _show_options("title")
        title = input("  > ").strip()
        if title.lower() == "quit":
            break

        print("  Experience Level")
        _show_options("formatted_experience_level")
        experience = input("  > ").strip()
        if experience.lower() == "quit":
            break

        print("  Work Type")
        if work_col:
            _show_options(work_col)
        work_type = input("  > ").strip()
        if work_type.lower() == "quit":
            break

        print("  Industry / Sector")
        if HAS_INDUSTRY and ind_name_col:
            _show_options(ind_name_col)
        industry = input("  > ").strip()
        if industry.lower() == "quit":
            break

        input_data = {}
        if work_col:
            input_data[work_col] = _encode(work_col, work_type or None)
        if "formatted_experience_level" in feature_cols:
            input_data["formatted_experience_level"] = _encode(
                "formatted_experience_level", experience or None)
        if HAS_INDUSTRY and ind_name_col in feature_cols:
            input_data[ind_name_col] = _encode(ind_name_col, industry or None)
        if "title" in feature_cols:
            input_data["title"] = _encode("title", title or None)

        input_df = pd.DataFrame([input_data])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        predicted = best_model.predict(input_df)[0]
        print(f"\n  +-----------------------------------------------+")
        print(f"  |  Predicted Salary :  ${predicted:>12,.0f} / year  |")
        print(f"  +-----------------------------------------------+")

        again = input("\n  Predict another? (y/n): ").strip().lower()
        if again != "y":
            break


# ════════════════════════════════════════════════════════════
#   RUN ALL PHASES
# ════════════════════════════════════════════════════════════

def run_all():
    _header("RUNNING ALL 5 PHASES")
    for fn in [run_phase1, run_phase2, run_phase3, run_phase4, run_phase5]:
        try:
            fn()
        except Exception as e:
            print(f"\n  [!]  Error in {fn.__name__}: {e}")
    print("\n  ALL PHASES COMPLETE!")


# ════════════════════════════════════════════════════════════
#   MAIN MENU
# ════════════════════════════════════════════════════════════

MENU = """
+======================================================+
|    JOB MARKET TREND ANALYZER  v2.0 -- MAIN MENU     |
+======================================================+
|  1  |  Phase 1 -- Data Loading, Cleaning & EDA      |
|  2  |  Phase 2 -- Skills Analysis                   |
|  3  |  Phase 3 -- Sector-wise Trend Analysis        |
|  4  |  Phase 4 -- Salary Analysis                   |
|  5  |  Phase 5 -- ML Salary Predictor               |
|  6  |  Interactive Salary Predictor                 |
|  7  |  Run ALL Phases (1 to 5)                      |
|  0  |  Exit                                         |
+======================================================+
"""

ACTIONS = {
    "1": run_phase1,
    "2": run_phase2,
    "3": run_phase3,
    "4": run_phase4,
    "5": run_phase5,
    "6": run_salary_predictor,
    "7": run_all,
}


def main():
    print("""
  +================================================+
  |   JOB MARKET TREND ANALYZER  v2.0             |
  |   LinkedIn Job Postings Dataset (2023-2024)    |
  +================================================+
    """)

    while True:
        print(MENU)
        choice = input("  Enter choice [0-7]: ").strip()

        if choice == "0":
            print("\n  Goodbye! Happy analyzing.\n")
            sys.exit(0)

        action = ACTIONS.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print("\n  [!]  Interrupted. Returning to menu ...")
            except Exception as e:
                print(f"\n  [X]  Unexpected error: {e}")
                import traceback; traceback.print_exc()
        else:
            print("  [!]  Invalid choice. Please enter 0-7.")

        input("\n  Press Enter to return to menu ...")


if __name__ == "__main__":
    main()
