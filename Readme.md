# DeepSPO

## Project Summary
**Intraday Market ML Pipeline** is an end-to-end project focused on building a machine-learning–ready intraday stock market dataset from raw exchange data. The project covers **data ingestion, cleaning, session-aware preprocessing, feature engineering, and modeling**, with a strong emphasis on avoiding data leakage and preserving true market structure.

The ultimate goal is to train robust ML models that learn from **real tradable market time**, not artificial or corrupted timestamps.

---

# Step 1: Intraday Data Cleaning & Preprocessing

## Data Source
- **Provider**: Alpaca Markets API (SIP feed)
- **Resolution**: 1‑minute OHLCV bars
- **Timezone (raw)**: UTC
- **Index**: Often returned as a MultiIndex (`symbol`, `timestamp`)

---

## Problems in Raw Data

- **Incorrect timezone** (UTC instead of New York time)
- **MultiIndex structure** breaks time-based operations
- **Off-hours & holiday trades** appear in SIP feed
- **Missing intraday minutes** (no trades in some minutes)
- **Overnight/weekend gaps** cause fake continuity if not handled correctly

---

## Cleaning Strategy (Solutions)

1. **Fix index structure**  
   Convert MultiIndex → pure `DatetimeIndex`.

2. **Timezone normalization**  
   Convert timestamps to `America/New_York`.

3. **Filter real market hours**  
   Keep only weekdays and regular session (09:30–16:00 ET).

4. **Session-aware processing**  
   Group data by trading day to prevent cross-day leakage.

5. **Intraday resampling (per day)**  
   Resample to a full 1‑minute grid using `asfreq()`.

6. **Safe filling**  
   - Forward-fill OHLC prices (no future leakage)
   - Zero-fill volume
   - Add `was_filled` flag for synthetic candles

7. **Drop invalid rows**  
   Remove leading NaNs before the first real trade.

---

## Handling Nights, Weekends & Holidays

- No rows are created for closed-market periods
- Holidays (e.g., Jan 1) naturally disappear if no session exists
- Timeline jumps directly from close → next open

This reflects **true tradable market time**.

---

## Result

The cleaned dataset:
- Contains only real trading minutes
- Has uniform 1‑minute spacing per session
- Is timezone-correct
- Is safe for ML and quantitative analysis

---

**This forms the first preprocessing stage of the project.**


