
import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

# ============================================================
# COINDCX RSI SCANNER — USDT FUTURES
# Conditions:
#   1. Daily RSI > 70 or < 30
#   2. Top/high-liquidity USDT perpetual coins by 24h quote volume
#   3. Average Daily Range
#   4. Historical Win Probability
# ============================================================

st.set_page_config(
    page_title="Daily RSI Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = "https://api.coindcx.com"
PUBLIC_BASE = "https://public.coindcx.com"

# ------------------------- STYLE ----------------------------

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top, #07182c 0%, #020914 48%, #01050b 100%);
        color: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        border: 1px solid #17395f;
        border-radius: 18px;
        padding: 20px 24px;
        background: linear-gradient(135deg, #06172c, #03101f);
        box-shadow: 0 10px 35px rgba(0,0,0,.30);
        margin-bottom: 16px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        letter-spacing: -.5px;
    }

    .hero p {
        margin: 4px 0 0;
        color: #9fb5ce;
        font-size: 16px;
    }

    .metric-card {
        border: 1px solid #17395f;
        border-radius: 14px;
        padding: 15px 17px;
        background: #061525;
        min-height: 100px;
    }

    .metric-title {
        color: #91a9c3;
        font-size: 13px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
        margin-top: 4px;
    }

    .metric-sub {
        color: #839ab4;
        font-size: 12px;
    }

    .section-red {
        border: 1px solid #9f1e35;
        border-radius: 15px;
        padding: 10px 14px 14px;
        background: linear-gradient(135deg, rgba(83,5,19,.45), rgba(8,15,27,.7));
        margin-top: 18px;
    }

    .section-green {
        border: 1px solid #087b51;
        border-radius: 15px;
        padding: 10px 14px 14px;
        background: linear-gradient(135deg, rgba(0,71,47,.40), rgba(8,15,27,.7));
        margin-top: 18px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        padding: 4px 2px 10px;
    }

    .red { color: #ff4d62; }
    .green { color: #19e39b; }
    .blue { color: #42a5ff; }
    .yellow { color: #ffd21c; }

    .small-note {
        color: #8299b3;
        font-size: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .prob-high { color: #19e39b; font-weight: 800; }
    .prob-mid { color: #ffd21c; font-weight: 800; }
    .prob-low { color: #ff5267; font-weight: 800; }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #205a96;
        background: #0b4fd3;
        color: white;
        font-weight: 700;
    }

    @media (max-width: 700px) {
        .hero h1 { font-size: 26px; }
        .block-container { padding-left: .65rem; padding-right: .65rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- API ------------------------------

session = requests.Session()
session.headers.update({"User-Agent": "DailyRSIScanner/1.0"})


@st.cache_data(ttl=60)
def get_exchange_info():
    """CoinDCX active USDT futures instruments."""
    url = f"{BASE}/exchange/v1/derivatives/futures/data/active_instruments"
    r = session.get(url, params={"margin_currency_short_name[]": "USDT"}, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_24h_tickers():
    """CoinDCX real-time futures prices and 24h statistics."""
    url = f"{PUBLIC_BASE}/market_data/v3/current_prices/futures/rt"
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def get_symbols(limit):
    instruments = get_exchange_info()
    ticker_payload = get_24h_tickers()
    prices = ticker_payload.get("prices", {})

    rows = []
    for pair in instruments:
        if pair not in prices:
            continue

        t = prices[pair]
        try:
            last_price = float(t.get("ls", 0))
            volume_base = float(t.get("v", 0))
            quote_volume = volume_base * last_price

            rows.append({
                "symbol": pair,
                "display_symbol": pair.replace("B-", "").replace("_USDT", "USDT"),
                "volume": quote_volume,
                "volume_base": volume_base,
                "price": last_price,
                "change": float(t.get("pc", 0)),
            })
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("volume", ascending=False).head(limit).reset_index(drop=True)

@st.cache_data(ttl=300)
def get_klines(symbol, timeframe="1D", limit=180):
    """CoinDCX Futures candles.

    CoinDCX REST futures candles support 1-minute, 5-minute, 1-hour and
    1-day resolutions. For the 2-minute strategy we fetch 1-minute candles
    and aggregate them into exact 2-minute OHLCV candles locally.
    """
    import time

    now = int(time.time())

    if timeframe == "2m":
        # Fetch enough 1-minute candles for RSI(34) and a useful short-term
        # backtest. 1000 is within the normal REST candle limit.
        one_min_limit = 1000
        from_ts = now - (one_min_limit + 20) * 60
        resolution = "1"
    else:
        from_ts = now - (limit + 10) * 24 * 60 * 60
        resolution = "1D"

    r = session.get(
        f"{PUBLIC_BASE}/market_data/candlesticks",
        params={
            "pair": symbol,
            "from": from_ts,
            "to": now,
            "resolution": resolution,
            "pcode": "f",
        },
        timeout=25,
    )
    r.raise_for_status()
    payload = r.json()
    raw = payload.get("data", [])

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["open_time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close", "open_time"])

    if timeframe == "2m":
        # Floor each 1-minute candle to a 2-minute bucket and aggregate OHLCV.
        df["bucket"] = df["open_time"].dt.floor("2min")
        grouped = df.groupby("bucket", as_index=False)
        counts = grouped.size().rename(columns={"size": "minute_count"})
        df = (
            grouped
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .rename(columns={"bucket": "open_time"})
            .merge(counts, on="open_time", how="left")
        )
        # Keep only complete 2-minute candles.
        df = df[df["minute_count"] == 2].drop(columns=["minute_count"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)

        # Remove the currently forming 2-minute candle.
        now_utc = pd.Timestamp.now(tz="UTC")
        if len(df) > 1:
            last_open = df.iloc[-1]["open_time"]
            if now_utc < last_open + pd.Timedelta(minutes=2):
                df = df.iloc[:-1].copy()

        return df.tail(limit).reset_index(drop=True)

    # Remove the currently forming daily candle.
    if len(df) > 1:
        now_utc = pd.Timestamp.now(tz="UTC")
        last_open = df.iloc[-1]["open_time"]
        if now_utc < last_open + pd.Timedelta(days=1):
            df = df.iloc[:-1].copy()

    return df.tail(limit).reset_index(drop=True)

def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # If loss is zero, RSI is 100.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    # If gain is zero, RSI is 0.
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)

    return rsi


def average_daily_range(df, days):
    if len(df) < days:
        return np.nan
    return float((df["high"] - df["low"]).tail(days).mean())


def historical_probability(
    df,
    rsi_period=34,
    range_days=20,
    lookahead=3,
    target_mult=1.0,
    stop_mult=1.0,
    max_setups=100,
):
    """Historical probability model for the selected timeframe."""

    if len(df) < max(range_days + rsi_period + lookahead + 10, 60):
        return np.nan, 0, 0, 0

    d = df.copy()
    d["rsi"] = rsi_wilder(d["close"], rsi_period)
    d["candle_range"] = d["high"] - d["low"]
    d["avg_range"] = d["candle_range"].rolling(range_days).mean()

    wins = 0
    losses = 0
    setups = 0
    start = max(range_days + rsi_period + 2, 2)

    for i in range(start, len(d) - lookahead):
        if setups >= max_setups:
            break

        prev_rsi = d.iloc[i - 1]["rsi"]
        cur_rsi = d.iloc[i]["rsi"]
        avg_rng = d.iloc[i - 1]["avg_range"]

        if pd.isna(prev_rsi) or pd.isna(cur_rsi) or pd.isna(avg_rng) or avg_rng <= 0:
            continue

        direction = None
        if prev_rsi <= 70 and cur_rsi > 70:
            direction = "SHORT"
        elif prev_rsi >= 30 and cur_rsi < 30:
            direction = "LONG"

        if direction is None:
            continue

        entry = float(d.iloc[i]["close"])
        target_dist = float(avg_rng) * target_mult
        stop_dist = float(avg_rng) * stop_mult

        if direction == "LONG":
            target = entry + target_dist
            stop = entry - stop_dist
        else:
            target = entry - target_dist
            stop = entry + stop_dist

        result = None
        for j in range(i + 1, i + 1 + lookahead):
            hi = float(d.iloc[j]["high"])
            lo = float(d.iloc[j]["low"])

            if direction == "LONG":
                hit_target = hi >= target
                hit_stop = lo <= stop
            else:
                hit_target = lo <= target
                hit_stop = hi >= stop

            if hit_target and hit_stop:
                result = "LOSS"
                break
            if hit_target:
                result = "WIN"
                break
            if hit_stop:
                result = "LOSS"
                break

        if result is not None:
            setups += 1
            if result == "WIN":
                wins += 1
            else:
                losses += 1

    if setups == 0:
        return np.nan, wins, losses, setups

    return wins / setups * 100.0, wins, losses, setups


def scan_one(
    row, timeframe, timeframe_label, rsi_period, range_days, lookahead,
    target_mult, stop_mult, max_setups, candle_limit
):
    symbol = row["symbol"]

    try:
        df = get_klines(symbol, timeframe, candle_limit)

        if len(df) < max(range_days + rsi_period + 5, 40):
            return None

        df["rsi"] = rsi_wilder(df["close"], rsi_period)
        df["candle_range"] = df["high"] - df["low"]
        df["avg_range"] = df["candle_range"].rolling(range_days).mean()

        last = df.iloc[-1]
        rsi = float(last["rsi"])
        price = float(last["close"])
        avg_range = float(last["avg_range"])

        if pd.isna(rsi) or pd.isna(avg_range):
            return None

        if rsi > 70:
            status = "RSI > 70"
            side = "SHORT"
        elif rsi < 30:
            status = "RSI < 30"
            side = "LONG"
        else:
            return None

        current_range = float(last["high"] - last["low"])
        range_pct = (current_range / price * 100) if price else np.nan

        prob, wins, losses, setups = historical_probability(
            df,
            rsi_period=rsi_period,
            range_days=range_days,
            lookahead=lookahead,
            target_mult=target_mult,
            stop_mult=stop_mult,
            max_setups=max_setups,
        )

        return {
            "Coin": str(row.get("display_symbol", symbol)),
            "RSI": rsi,
            "Signal": status,
            "Side": side,
            "Price": price,
            "24h Volume": float(row["volume"]),
            "Avg Range": avg_range,
            "Current Range": current_range,
            "Range %": range_pct,
            "Win Probability": prob,
            "Wins": wins,
            "Losses": losses,
            "Setups": setups,
            "24h Change %": float(row["change"]),
            "Timeframe": timeframe_label,
        }

    except Exception:
        return None


# ------------------------- HEADER ---------------------------

st.markdown("""
<div class="hero">
    <h1>📊 CoinDCX RSI Scanner</h1>
    <p>High Liquidity Futures • Select 2M or 1D • RSI 34 Extremes • Historical Win Probability</p>
</div>
""", unsafe_allow_html=True)

# ------------------------- CONTROLS -------------------------

strategy = st.selectbox(
    "Strategy",
    ["2M RSI 34", "1D RSI 34"],
    index=1,
    help="2M uses CoinDCX 1-minute futures candles aggregated into 2-minute candles."
)

if strategy == "2M RSI 34":
    timeframe = "2m"
    timeframe_label = "2M"
    candle_limit = 1000
else:
    timeframe = "1D"
    timeframe_label = "1D"
    candle_limit = 180

rsi_period = 34

c1, c2, c3, c4, c5 = st.columns([1.3, 1, .8, 1, 1.3])

with c1:
    coin_limit = st.selectbox("No. of Coins", [30, 50, 100, 150, 200], index=4)

with c2:
    st.number_input("RSI Period", min_value=34, max_value=34, value=34, disabled=True)

with c3:
    range_days = st.selectbox("Avg Range", [7, 14, 20, 30], index=2)

with c4:
    lookahead = st.selectbox("Win Window", [1, 3, 5], index=1)

with c5:
    min_volume_m = st.number_input(
        "Min 24h Turnover ($M)",
        min_value=0.0,
        value=50.0,
        step=10.0
    )

c6, c7, c8, c9 = st.columns([1.3, 1.2, 1.2, 1.2])

with c6:
    target_mult = st.number_input("Target × Avg Range", 0.25, 5.0, 1.0, 0.25)

with c7:
    stop_mult = st.number_input("Stop × Avg Range", 0.25, 5.0, 1.0, 0.25)

with c8:
    max_setups = st.selectbox("Backtest Setups", [30, 50, 75, 100], index=3)

with c9:
    auto_refresh = st.toggle("🔄 Auto Refresh", value=False)

scan = st.button("🔎  SCAN NOW", use_container_width=True, type="primary")

if auto_refresh:
    st_autorefresh(interval=5 * 60 * 1000, key="daily_rsi_auto_refresh")
    st.markdown(
        "<div class='small-note'>🔄 Auto-refresh is ON • every 5 minutes</div>",
        unsafe_allow_html=True
    )

# Session state
if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

if scan or st.session_state.scan_result is None or st.session_state.get("last_strategy") != strategy:
    with st.spinner(f"Scanning CoinDCX Futures using {strategy}..."):
        universe = get_symbols(coin_limit)

        if universe.empty:
            st.error("Could not load CoinDCX Futures symbols.")
            st.stop()

        universe = universe[universe["volume"] >= min_volume_m * 1_000_000].copy()

        results = []
        workers = min(12, max(4, len(universe)))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    scan_one,
                    row,
                    timeframe,
                    timeframe_label,
                    rsi_period,
                    range_days,
                    lookahead,
                    target_mult,
                    stop_mult,
                    max_setups,
                    candle_limit,
                )
                for _, row in universe.iterrows()
            ]

            for f in as_completed(futures):
                item = f.result()
                if item:
                    results.append(item)

        result_df = pd.DataFrame(results)

        if not result_df.empty:
            result_df = result_df.sort_values(
                ["Win Probability", "RSI"],
                ascending=[False, False],
                na_position="last"
            ).reset_index(drop=True)

        st.session_state.scan_result = {
            "df": result_df,
            "scanned": len(universe),
            "updated": datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "strategy": strategy,
        }
        st.session_state.last_strategy = strategy

data = st.session_state.scan_result
df = data["df"]

above = df[df["RSI"] > 70].copy() if not df.empty else pd.DataFrame()
below = df[df["RSI"] < 30].copy() if not df.empty else pd.DataFrame()
valid_prob = df["Win Probability"].dropna() if not df.empty else pd.Series(dtype=float)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💧 Coins Scanned</div>
        <div class="metric-value blue">{data["scanned"]}</div>
        <div class="metric-sub">Top liquidity by estimated 24h USDT turnover</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📈 RSI Above 70</div>
        <div class="metric-value red">{len(above)}</div>
        <div class="metric-sub">Potential overbought / short setups</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📉 RSI Below 30</div>
        <div class="metric-value green">{len(below)}</div>
        <div class="metric-sub">Potential oversold / long setups</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    avg_prob = valid_prob.mean() if len(valid_prob) else np.nan
    prob_text = f"{avg_prob:.1f}%" if not pd.isna(avg_prob) else "—"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Avg Win Probability</div>
        <div class="metric-value yellow">{prob_text}</div>
        <div class="metric-sub">Historical {strategy} setup estimate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="small-note">Data source: CoinDCX Futures public market-data API</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Strategy: {strategy} • RSI 34 • Last updated: {data['updated']} • "
    f"CoinDCX USDT Futures • Completed {timeframe_label} candles used."
)

def show_table(x, section_class, title, emoji):
    if x.empty:
        st.markdown(f"""
        <div class="{section_class}">
            <div class="section-title">{emoji} {title} <span class="small-note">No coins</span></div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="{section_class}">
        <div class="section-title">{emoji} {title}
        <span style="float:right">{len(x)} Coins</span></div>
    """, unsafe_allow_html=True)

    view = x.copy()
    view["Price"] = view["Price"].map(lambda v: f"{v:,.8f}".rstrip("0").rstrip("."))
    view["24h Volume"] = view["24h Volume"].map(
        lambda v: f"${v/1e9:.2f}B" if v >= 1e9 else f"${v/1e6:.1f}M"
    )
    view["Avg Range"] = view["Avg Range"].map(lambda v: f"{v:,.8f}".rstrip("0").rstrip("."))
    view["Current Range"] = view["Current Range"].map(lambda v: f"{v:,.8f}".rstrip("0").rstrip("."))
    view["Range %"] = view["Range %"].map(lambda v: f"{v:.2f}%")
    view["Win Probability"] = view["Win Probability"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.1f}%"
    )
    view["RSI"] = view["RSI"].map(lambda v: f"{v:.1f}")
    view["24h Change %"] = view["24h Change %"].map(lambda v: f"{v:+.2f}%")

    cols = [
        "Coin", "RSI", "Side", "Price", "24h Volume",
        "Avg Range", "Current Range", "Range %",
        "Win Probability", "Wins", "Losses", "Setups", "24h Change %"
    ]

    st.dataframe(
        view[cols],
        use_container_width=True,
        hide_index=True,
        height=min(600, 80 + len(view) * 38),
    )

    st.markdown("</div>", unsafe_allow_html=True)


show_table(above, "section-red", f"{strategy}  •  RSI ABOVE 70", "🔥")
show_table(below, "section-green", f"{strategy}  •  RSI BELOW 30", "🌱")

# ------------------------- EXPLANATION ----------------------

with st.expander("🟦 CoinDCX data source"):
    st.markdown("""
The scanner uses CoinDCX Futures public market-data endpoints for:
- Active USDT futures instruments
- Current futures prices and 24h statistics
- Futures candlesticks

No CoinDCX API key is required for these public market-data calls.

**2M strategy:** CoinDCX's REST futures candle endpoint provides 1-minute
candles, so the scanner combines consecutive 1-minute candles into 2-minute
OHLCV candles before calculating RSI 34.
""")

with st.expander("🎯 How Win Probability is calculated"):
    st.markdown(f"""
**Historical probability — not a guaranteed future win rate.**

- Current strategy: **{strategy}**.
- RSI period is fixed at **34**.
- A setup is created when RSI **crosses above 70** or **crosses below 30**.
- RSI > 70 = hypothetical **SHORT**.
- RSI < 30 = hypothetical **LONG**.
- Entry = close of the RSI signal candle.
- Target = **{target_mult:.2f} × {range_days} candle Average Range**.
- Stop = **{stop_mult:.2f} × {range_days} candle Average Range**.
- Outcome is checked over the next **{lookahead} completed {timeframe_label} candles**.
- If target and stop occur in the same candle, it is conservatively counted as a **loss**.
- The table shows **Wins / Losses / Setups** so small samples are visible.
""")

with st.expander("📐 Average Range"):
    st.markdown(
        f"**{range_days} {timeframe_label} candle Average Range = average(High − Low) over the previous "
        f"{range_days} completed {timeframe_label} candles.**"
    )

st.markdown("""
<div class="small-note" style="margin-top:18px;">
⚠️ Win Probability is a historical backtest statistic. It is not a prediction or guarantee of profit.
Crypto futures involve substantial risk.
</div>
""", unsafe_allow_html=True)
