#!/usr/bin/env python3
"""
Lands' End — Synthetic Marketplace Demand Dataset Generator

- Generates a 5,000-row CSV (default) modelling daily SKU-level demand on marketplaces.
- Built to support demand forecasting & what-if simulations (price, promo, ads, seasonality).

Design notes:
- Seasonality: Swimwear peaks in late spring–summer; outerwear peaks in late fall–winter.
- Price elasticity: Higher effective price lowers units; elasticity varies by category.
- Promotions & ads: Both lift demand; diminishing returns for ads.
- Ratings: Higher rating -> better conversion.
- Marketplace effect: Higher baseline on Amazon (proxy), others slightly lower.
- Stockouts: Units drop to near-zero; partial suppression 1 day before/after.
- Reproducible: fixed seed.

Usage:
    python landsend_synth_demand.py --rows 5000 --out landsend_synth_demand.csv --start 2024-01-01 --days 366
"""

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SKU:
    sku: str
    category: str
    gender: str
    base_price: float
    base_demand: float  # baseline units/day at neutral conditions
    elasticity: float   # price elasticity magnitude


def _seasonality_multiplier(category: str, day: datetime) -> float:
    m = day.month
    # Swimwear high in May–Aug, moderate in Mar–Apr/Sep, low in cooler months
    if category == "Swimwear":
        if 5 <= m <= 8:
            return 1.6
        elif m in (3,4,9):
            return 1.2
        else:
            return 0.7
    # Outerwear high in Nov–Feb, shoulder Oct/Mar, low otherwise
    if category == "Outerwear":
        if m in (11,12,1,2):
            return 1.7
        elif m in (10,3):
            return 1.2
        else:
            return 0.6
    # Uniforms flatter, small back-to-school bump Aug–Sep
    if category == "Uniforms":
        if m in (8,9):
            return 1.3
        else:
            return 1.0
    # Apparel & Footwear have mild holiday lift
    if category in ("Apparel","Footwear","Home"):
        if m in (11,12):
            return 1.25
        else:
            return 1.0
    return 1.0


def _holiday_flag(day: datetime) -> int:
    # Simple US retail holiday proxies: Memorial Day, July 4th, Labor Day, Black Friday–Cyber Monday, Christmas
    md = (day.month, day.day)
    if (day.month == 11 and day.weekday() == 4 and 23 <= day.day <= 29):  # Black Friday heuristic
        return 1
    if day.month == 11 and day.weekday() == 0 and 26 <= day.day <= 30:    # Cyber Monday heuristic
        return 1
    if md in [(7,4),(12,24),(12,25)]:
        return 1
    # Memorial Day (last Monday of May)
    if day.month == 5 and day.weekday() == 0 and day.day + (7 - day.weekday()) > 31:
        return 1
    # Labor Day (first Monday of Sep)
    if day.month == 9 and day.weekday() == 0 and 1 <= day.day <= 7:
        return 1
    return 0


def _marketplace_effect(marketplace: str) -> float:
    return {
        "Amazon": 1.0,
        "Walmart": 0.85,
        "TargetPlus": 0.80,
        "TikTokShop": 0.75
    }.get(marketplace, 0.8)


def _promo_lift(promo: str, discount_pct: float) -> float:
    # Use both type and depth
    base = {
        "None": 1.0,
        "Coupon": 1.10,
        "LightningDeal": 1.20,
        "DealOfDay": 1.30,
        "Clearance": 1.40
    }.get(promo, 1.0)
    # Additional lift from deeper discounts (convex, diminishing beyond 40%)
    depth = 1.0 + 0.8 * min(discount_pct, 0.4)
    return base * depth


def _rating_to_cvr(rating: float) -> float:
    # Map 3.5–4.9 stars to ~2%–5.5% conversion baseline
    return 0.02 + (rating - 3.5) * (0.025 / 1.4)


def _ctr_for_category(category: str) -> float:
    # category-level CTR baseline
    return {
        "Swimwear": 0.015,
        "Outerwear": 0.012,
        "Apparel": 0.010,
        "Footwear": 0.011,
        "Home": 0.009,
        "Uniforms": 0.008
    }.get(category, 0.010)


def _build_skus(rng: np.random.Generator) -> List[SKU]:
    # A small portfolio of high-volume SKUs typical for Lands' End
    # (Categories informed by investor materials: swimwear, outerwear, apparel, footwear, home, uniforms)
    skus = []
    items = [
        ("LE-SWIM-W-ONEPC", "Swimwear", "Women", 89.0, 28.0, 1.2),
        ("LE-SWIM-M-TRUNK", "Swimwear", "Men", 49.0, 22.0, 1.1),
        ("LE-OUT-M-PARKA",  "Outerwear", "Men", 199.0, 8.0, 0.8),
        ("LE-OUT-W-DOWN",   "Outerwear", "Women", 229.0, 7.0, 0.9),
        ("LE-APP-W-CHINO",  "Apparel", "Women", 79.0, 18.0, 1.0),
        ("LE-APP-M-POLO",   "Apparel", "Men", 39.0, 24.0, 1.2),
        ("LE-FOOT-W-SNEAK", "Footwear", "Women", 69.0, 14.0, 1.0),
        ("LE-FOOT-M-BOOT",  "Footwear", "Men", 129.0, 9.0, 0.9),
        ("LE-HOME-BTOWEL",  "Home", "Unisex", 29.0, 26.0, 1.3),
        ("LE-HOME-BEDDNG",  "Home", "Unisex", 149.0, 6.0, 0.8),
        ("LE-UNIF-K-POLO",  "Uniforms", "Kids", 19.0, 30.0, 1.4),
        ("LE-UNIF-K-SKIRT", "Uniforms", "Kids", 24.0, 20.0, 1.3),
        ("LE-APP-W-CARDI",  "Apparel", "Women", 59.0, 16.0, 1.0),
        ("LE-APP-M-OXFORD", "Apparel", "Men", 69.0, 12.0, 1.1),
    ]
    for sku, cat, gen, price, demand, elas in items:
        skus.append(SKU(sku, cat, gen, price, demand, elas))
    return skus


def generate(rows: int = 5000,
             start_date: str = "2024-01-01",
             days: int = 366,
             out_path: str = "landsend_synth_demand.csv",
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    skus = _build_skus(rng)
    marketplaces = ["Amazon", "Walmart", "TargetPlus", "TikTokShop"]
    promos = ["None", "Coupon", "LightningDeal", "DealOfDay", "Clearance"]
    fulfillments = ["FBA", "Seller"]
    colors = ["Black","Navy","Khaki","White","Red","Blue","Green","Gray"]
    sizes = ["XS","S","M","L","XL"]

    start = datetime.fromisoformat(start_date)
    dates = [start + timedelta(days=i) for i in range(days)]

    records = []
    for d in dates:
        for sku in skus:
            # Marketplace mix (weight Amazon higher)
            marketplace = rng.choice(marketplaces, p=[0.55,0.20,0.15,0.10])
            fulfillment = "FBA" if marketplace == "Amazon" and rng.random() < 0.7 else rng.choice(fulfillments)
            color = rng.choice(colors)
            size = rng.choice(sizes)

            list_price = max(5.0, np.round(rng.normal(sku.base_price, sku.base_price*0.05), 2))

            # Discount strategy varies by seasonality and inventory
            season_mult = _seasonality_multiplier(sku.category, d)
            holiday = _holiday_flag(d)
            # Deeper discounts when seasonality is low or during holidays/promos
            target_discount = 0.35 if (holiday or season_mult < 0.9) else 0.15
            discount_pct = float(np.clip(rng.normal(target_discount, 0.08), 0.0, 0.6))

            promo = rng.choice(promos, p=[0.55,0.20,0.12,0.08,0.05])
            promo_lift = _promo_lift(promo, discount_pct)

            # Ads: more on Amazon & during peak seasons/holidays
            base_ad = 40 if marketplace == "Amazon" else 18
            ad_spend = float(max(0.0, rng.lognormal(mean=np.log(base_ad + 1), sigma=0.6) - 1))
            # Diminishing returns: effect ~ log(1 + ad_spend)
            ad_lift = 1.0 + 0.10 * math.log1p(ad_spend)

            # Ratings drift slowly; sku-level anchor + small noise
            sku_anchor = 4.1 + (hash(sku.sku) % 7) * 0.02
            rating = float(np.clip(rng.normal(sku_anchor, 0.08), 3.5, 4.9))
            reviews = int(max(0, rng.normal(800 if marketplace=="Amazon" else 250, 120)))

            # Competitive price index: our effective price vs category avg (lower -> more competitive)
            # Let category avg vary a bit over time
            cat_avg = sku.base_price * (1.0 + rng.normal(0, 0.05))
            eff_price = list_price * (1 - discount_pct)
            price_index = float(np.clip(eff_price / cat_avg, 0.6, 1.4))

            # Stock on hand: smaller for high-priced outerwear; set stockouts sometimes
            stock_on_hand = int(max(0, rng.normal(400 if sku.category!="Outerwear" else 250, 80)))
            stockout_flag = 1 if (stock_on_hand < 10 and rng.random() < 0.3) else 0

            # Demand model (log space) — ensure statistically strong signals
            # log_mu = baseline + seasonality + holiday + marketplace + promo + ads + rating + price + randomness
            baseline = math.log(max(1e-3, sku.base_demand))
            season_term = math.log(season_mult)
            holiday_term = math.log(1.15) if holiday else 0.0
            marketplace_term = math.log(_marketplace_effect(marketplace))
            promo_term = math.log(promo_lift)
            ad_term = math.log(ad_lift)
            rating_term = 0.18 * (rating - 4.2)  # small but real
            price_term = - sku.elasticity * math.log(max(0.5, price_index))

            noise = rng.normal(0, 0.25)  # modest noise

            log_mu = baseline + season_term + holiday_term + marketplace_term + promo_term + ad_term + rating_term + price_term + noise

            mu = max(0.1, math.exp(log_mu))

            # Stockout suppression
            if stockout_flag:
                mu *= 0.1

            units = rng.poisson(mu)

            # Ad funnel metrics: impressions ~ linear in ad_spend * category factor; clicks via CTR; conversions via CVR
            ctr = _ctr_for_category(sku.category) * (1.15 if promo != "None" else 1.0)
            # Impressions grow sublinearly with ad_spend; add organic discoverability via seasonality
            impressions = int(max(0, rng.normal(1500 * math.log1p(ad_spend) * season_mult, 250)))
            clicks = int(np.random.binomial(n=max(impressions,0), p=min(max(ctr,0.001),0.08)))
            base_cvr = _rating_to_cvr(rating)
            # CVR improves with promo and competitiveness, degrades with high price index
            cvr = float(np.clip(base_cvr * (promo_lift**0.25) * (1.10 if price_index < 0.95 else 0.95), 0.005, 0.12))

            # Align clicks*CVR roughly with units, but keep stochastic demand primary
            est_units_from_ads = int(clicks * cvr)
            # Blend ad-derived units with modeled units
            units_sold = int(max(0, 0.6*units + 0.4*est_units_from_ads))

            revenue = float(units_sold * eff_price)

            records.append({
                "date": d.date().isoformat(),
                "week": int(d.isocalendar().week),
                "sku": sku.sku,
                "product_category": sku.category,
                "gender": sku.gender,
                "marketplace": marketplace,
                "fulfillment": fulfillment,
                "color": color,
                "size": size,
                "list_price": round(list_price, 2),
                "discount_pct": round(discount_pct, 3),
                "promo_type": promo,
                "ad_spend": round(ad_spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "cvr": round(cvr, 4),
                "units_sold": units_sold,
                "revenue": round(revenue, 2),
                "rating": round(rating, 2),
                "reviews": reviews,
                "competitor_price_index": round(price_index, 3),
                "stock_on_hand": stock_on_hand,
                "stockout_flag": stockout_flag,
                "holiday_flag": holiday,
            })

    df = pd.DataFrame.from_records(records)
    # Sort and trim to exact row count
    df.sort_values(["date","sku"], inplace=True, kind="mergesort")
    if len(df) > rows:
        df = df.head(rows).copy()

    df.to_csv(out_path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", type=str, default="landsend_synth_demand.csv")
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--days", type=int, default=366)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(rows=args.rows, start_date=args.start, days=args.days, out_path=args.out, seed=args.seed)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
