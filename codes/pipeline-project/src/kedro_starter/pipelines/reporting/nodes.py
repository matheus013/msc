from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---- helpers ---------------------------------------------------------------
def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        # robust date parsing
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
    return df

def _safe_savefig(path: Path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    # tight bbox + dpi alto para ficar nítido em slides
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)

# ---- main ------------------------------------------------------------------
def generate_sales_reports(vendas_enrich: pd.DataFrame, reports_dir: str) -> None:
    """
    Create presentation-ready charts and a markdown summary from vendas_enrich.

    Expected columns (as defined in your enrich step):
      date, qty, rev_gross, rev_net, delivery_channel, capture_channel, cycle,
      rev_id, prod_id, city, state, segment, att_type, branch, plaza, gr,
      prod_group, prod_category, prod_subcategory, prod_condition, prod_brand
    """
    outdir = Path(reports_dir)
    df = vendas_enrich.copy()

    # --- clean & derive ------------------------------------------------------
    df = _ensure_datetime(df, "date")
    df["qty"] = pd.to_numeric(df.get("qty"), errors="coerce")
    df["rev_net"] = pd.to_numeric(df.get("rev_net"), errors="coerce")
    df["rev_gross"] = pd.to_numeric(df.get("rev_gross"), errors="coerce")

    # guard against empty/invalid rows
    df = df.dropna(subset=["date", "rev_net", "qty"], how="any")

    # add month for rollups
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # --- KPIs ----------------------------------------------------------------
    kpis = {
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "orders": int(len(df)),
        "unique_resellers": int(df["rev_id"].nunique() if "rev_id" in df else 0),
        "unique_products": int(df["prod_id"].nunique() if "prod_id" in df else 0),
        "qty_total": float(df["qty"].sum()),
        "rev_net_total": float(df["rev_net"].sum()),
        "rev_gross_total": float(df["rev_gross"].sum() if "rev_gross" in df else 0.0),
        "avg_ticket": float((df["rev_net"].sum() / max(df["qty"].sum(), 1.0))),
    }
    # save KPIs CSV
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([kpis]).to_csv(outdir / "kpis_summary.csv", index=False)

    # --- Charts --------------------------------------------------------------
    # 1) Revenue over time (monthly)
    m = df.groupby("month", as_index=False)["rev_net"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m["month"], m["rev_net"], marker="o")
    ax.set_title("Net revenue by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net revenue")
    ax.grid(True, alpha=0.3)
    _safe_savefig(outdir / "rev_by_month.png", fig)

    # 2) Top products by net revenue
    if "prod_id" in df.columns:
        top_prod = (df.groupby("prod_id", as_index=False)["rev_net"]
                    .sum().sort_values("rev_net", ascending=False).head(15))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_prod["prod_id"].astype(str), top_prod["rev_net"])
        ax.invert_yaxis()
        ax.set_title("Top products by net revenue")
        ax.set_xlabel("Net revenue")
        ax.set_ylabel("Product")
        _safe_savefig(outdir / "top_products.png", fig)

    # 3) Revenue by state (or city fallback)
    geo_col = "state" if "state" in df.columns else ("city" if "city" in df.columns else None)
    if geo_col:
        geo_rev = (df.groupby(geo_col, as_index=False)["rev_net"]
                   .sum().sort_values("rev_net", ascending=False).head(15))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(geo_rev[geo_col].astype(str), geo_rev["rev_net"])
        ax.invert_yaxis()
        ax.set_title(f"Net revenue by {geo_col}")
        ax.set_xlabel("Net revenue")
        ax.set_ylabel(geo_col.capitalize())
        _safe_savefig(outdir / f"rev_by_{geo_col}.png", fig)

    # 4) Revenue by segment
    if "segment" in df.columns:
        seg = (df.groupby("segment", as_index=False)["rev_net"]
               .sum().sort_values("rev_net", ascending=False))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(seg["segment"].astype(str), seg["rev_net"])
        ax.set_title("Net revenue by segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Net revenue")
        ax.tick_params(axis="x", rotation=30)
        _safe_savefig(outdir / "rev_by_segment.png", fig)

    # 5) Channel mix (delivery / capture)
    if "delivery_channel" in df.columns:
        dch = df["delivery_channel"].value_counts(dropna=False).head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(dch.index.astype(str), dch.values)
        ax.set_title("Orders by delivery channel")
        ax.set_xlabel("Delivery channel")
        ax.set_ylabel("Orders")
        ax.tick_params(axis="x", rotation=20)
        _safe_savefig(outdir / "orders_by_delivery_channel.png", fig)

    if "capture_channel" in df.columns:
        cch = df["capture_channel"].value_counts(dropna=False).head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cch.index.astype(str), cch.values)
        ax.set_title("Orders by capture channel")
        ax.set_xlabel("Capture channel")
        ax.set_ylabel("Orders")
        ax.tick_params(axis="x", rotation=20)
        _safe_savefig(outdir / "orders_by_capture_channel.png", fig)

    # 6) Average ticket by month
    m2 = df.groupby("month", as_index=False).agg(rev_net=("rev_net", "sum"),
                                                 qty=("qty", "sum"))
    m2["avg_ticket"] = m2["rev_net"] / m2["qty"].replace(0, pd.NA)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m2["month"], m2["avg_ticket"], marker="o")
    ax.set_title("Average ticket by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg ticket (rev_net / qty)")
    ax.grid(True, alpha=0.3)
    _safe_savefig(outdir / "avg_ticket_by_month.png", fig)

    # --- Markdown report (high-level) ----------------------------------------
    md = outdir / "report_summary.md"
    md.write_text(
f"""# Sales report (vendas_enrich)

**Period:** {kpis['date_min'].date() if pd.notna(kpis['date_min']) else '-'} → {kpis['date_max'].date() if pd.notna(kpis['date_max']) else '-'}  
**Orders:** {kpis['orders']:,}  
**Unique resellers:** {kpis['unique_resellers']:,}  
**Unique products:** {kpis['unique_products']:,}

**Total net revenue:** {kpis['rev_net_total']:,.2f}  
**Total gross revenue:** {kpis['rev_gross_total']:,.2f}  
**Total quantity:** {kpis['qty_total']:,.0f}  
**Average ticket:** {kpis['avg_ticket']:,.2f}

## Figures
- rev_by_month.png
- top_products.png
- rev_by_state / rev_by_city.png
- rev_by_segment.png
- orders_by_delivery_channel.png
- orders_by_capture_channel.png
- avg_ticket_by_month.png
""",
        encoding="utf-8",
    )
