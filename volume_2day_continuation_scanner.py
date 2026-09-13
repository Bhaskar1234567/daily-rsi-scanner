import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ============================================================
# CoinDCX Futures - 2-Day Continuation + Volume Spike Scanner
# This is a SEPARATE scanner. It does not modify your RSI scanner.
# ============================================================

BASE = "https://api.coindcx.com"
PUBLIC_BASE = "https://public.coindcx.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CoinDCX-2Day-Volume-Scanner/1.0"})
TIMEOUT = 10
WORKERS = 12

TIMEFRAME_OPTIONS = {
    "1m": ("1", 60),
    "3m": ("3", 180),
    "5m": ("5", 300),
    "15m": ("15", 900),
    "30m": ("30", 1800),
    "1H": ("60", 3600),
    "4H": ("240", 14400),
}

# -----------------------------
# CoinDCX data
# -----------------------------

@st.cache_data(ttl=120, show_spinner=False)
def get_symbols(limit=200):
    url = f"{BASE}/exchange/v1/derivatives/futures/data/active_instruments"
    r = SESSION.get(
        url,
        params={"margin_currency_short_name[]": "USDT"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        data = data.get("data") or data.get("instruments") or data.get("result") or []

    pairs = []
    for item in data:
        if isinstance(item, str):
            pair = item
        elif isinstance(item, dict):
            pair = item.get("pair") or item.get("symbol")
        else:
            pair = None
        if pair and str(pair).endswith("_USDT"):
            pairs.append(str(pair))

    if not pairs:
        raise RuntimeError("CoinDCX returned no active USDT futures pairs.")

    # Current futures prices + 24h volume.
    pr = SESSION.get(
        f"{PUBLIC_BASE}/market_data/v3/current_prices/futures/rt",
        timeout=TIMEOUT,
    )
    pr.raise_for_status()
    pdata = pr.json()
    prices = pdata.get("prices", {}) if isinstance(pdata, dict) else {}

    rows = []
    for pair in pairs:
        obj = prices.get(pair, {})
        if not isinstance(obj, dict):
            obj = {}
        try:
            last = float(obj.get("ls") or 0)
        except Exception:
            last = 0.0
        try:
            vol = float(obj.get("v") or 0)
        except Exception:
            vol = 0.0
        rows.append({
            "Pair": pair,
            "Coin": pair.replace("B-", "").replace("_USDT", ""),
            "24h Volume": vol,
            "Last": last,
            "Turnover": vol * last,
        })

    return (
        pd.DataFrame(rows)
        .drop_duplicates("Pair")
        .sort_values("Turnover", ascending=False, na_position="last")
        .head(int(limit))
        .reset_index(drop=True)
    )


def parse_candles(data):
    rows = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected candle response from CoinDCX.")

    out = []
    for x in rows:
        if isinstance(x, dict):
            ts = x.get("time") or x.get("timestamp") or x.get("t")
            o = x.get("open") or x.get("o")
            h = x.get("high") or x.get("h")
            l = x.get("low") or x.get("l")
            c = x.get("close") or x.get("c")
            v = x.get("volume") or x.get("v") or 0
        else:
            if len(x) < 5:
                continue
            ts, o, h, l, c = x[:5]
            v = x[5] if len(x) > 5 else 0

        try:
            ts = float(ts)
            if ts > 10_000_000_000:
                ts /= 1000
            out.append([
                pd.to_datetime(ts, unit="s", utc=True),
                float(o), float(h), float(l), float(c), float(v)
            ])
        except Exception:
            continue

    if not out:
        raise RuntimeError("No usable candles returned.")

    return pd.DataFrame(
        out, columns=["time", "open", "high", "low", "close", "volume"]
    ).sort_values("time").drop_duplicates("time").reset_index(drop=True)


@st.cache_data(ttl=45, show_spinner=False)
def get_klines(pair, timeframe, limit=220):
    now = int(time.time())
    url = f"{PUBLIC_BASE}/market_data/candlesticks"

    if timeframe == "3m":
        # CoinDCX 3m is safely built from complete 1m candles.
        raw_limit = max(limit * 3 + 20, 250)
        r = SESSION.get(
            url,
            params={
                "pair": pair,
                "from": now - raw_limit * 60,
                "to": now,
                "resolution": "1",
                "pcode": "f",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        raw = parse_candles(r.json())
        raw["bucket"] = raw["time"].dt.floor("3min")
        df = (
            raw.groupby("bucket", sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                count=("close", "size"),
            )
            .reset_index()
            .rename(columns={"bucket": "time"})
        )
        df = df[df["count"] == 3].drop(columns="count")
        df = df[df["time"] + pd.Timedelta(minutes=3) <= pd.Timestamp.now(tz="UTC")]
        return df.tail(limit).reset_index(drop=True)

    if timeframe == "1D":
        r = SESSION.get(
            url,
            params={
                "pair": pair,
                "from": now - (limit + 5) * 86400,
                "to": now,
                "resolution": "1D",
                "pcode": "f",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        df = parse_candles(r.json())
        today = pd.Timestamp.now(tz="UTC").floor("D")
        return df[df["time"] < today].tail(limit).reset_index(drop=True)

    if timeframe not in TIMEFRAME_OPTIONS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    resolution, seconds = TIMEFRAME_OPTIONS[timeframe]
    # Extra history prevents losing bars at the start of a scan.
    r = SESSION.get(
        url,
        params={
            "pair": pair,
            "from": now - (limit + 5) * seconds,
            "to": now,
            "resolution": resolution,
            "pcode": "f",
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    df = parse_candles(r.json())

    # Never use the currently forming candle.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=seconds)
    df = df[df["time"] <= cutoff]
    return df.tail(limit).reset_index(drop=True)


# -----------------------------
# 2-Day continuation
# -----------------------------

def two_day_continuation(daily):
    """Uses completed daily candles only.

    BUY:
      Day 1 close > previous day high
      Day 2 close > Day 1 high

    SELL:
      Day 1 close < previous day low
      Day 2 close < Day 1 low
    """
    if len(daily) < 3:
        return {"signal": "NONE"}

    prev = daily.iloc[-3]
    day1 = daily.iloc[-2]
    day2 = daily.iloc[-1]

    bull = (
        day1["close"] > prev["high"]
        and day2["close"] > day1["high"]
    )
    bear = (
        day1["close"] < prev["low"]
        and day2["close"] < day1["low"]
    )

    if bull:
        signal = "BUY"
    elif bear:
        signal = "SELL"
    else:
        signal = "NONE"

    return {
        "signal": signal,
        "prev_high": float(prev["high"]),
        "prev_low": float(prev["low"]),
        "day1_high": float(day1["high"]),
        "day1_low": float(day1["low"]),
        "day1_close": float(day1["close"]),
        "day2_close": float(day2["close"]),
        "day2_volume": float(day2["volume"]),
        "day1_volume": float(day1["volume"]),
        "volume_ratio": float(day2["volume"] / day1["volume"]) if day1["volume"] > 0 else 0.0,
        "day1_time": day1["time"],
        "day2_time": day2["time"],
    }


# -----------------------------
# Volume spike / dry-up / re-expansion
# -----------------------------

def volume_pattern(
    df,
    first_spike_x=10.0,
    second_expand_x=2.0,
    low_volume_x=0.60,
    low_bars=5,
    slowdown_bars=6,
    consolidation_bars=6,
    max_consolidation_pct=3.0,
    breakout_buffer_pct=0.10,
):
    """Detect:

    BIG SPIKE -> VOLUME DRY-UP -> PRICE SLOWS -> CONSOLIDATION
    -> VOLUME RE-EXPANDS -> BREAKOUT

    First spike is measured against the preceding 20-candle median volume.
    Re-expansion is measured against the dry-up average volume.
    """
    need = max(80, low_bars + slowdown_bars + consolidation_bars + 20)
    if len(df) < need:
        return None

    x = df.copy().reset_index(drop=True)
    x["range_pct"] = (x["high"] - x["low"]) / x["close"].replace(0, np.nan) * 100.0
    x["baseline_vol"] = x["volume"].rolling(20).median().shift(1)
    x["vol_x"] = x["volume"] / x["baseline_vol"].replace(0, np.nan)

    # Search for a second volume expansion in the last 3 completed candles.
    expansion_idx = None
    expansion_ratio = 0.0
    dry_mean_for_expansion = 0.0

    for idx in range(len(x) - 1, max(30, len(x) - 4), -1):
        dry_start = idx - low_bars
        if dry_start < 20:
            continue
        dry = x.iloc[dry_start:idx]
        if len(dry) < low_bars:
            continue
        dry_mean = float(dry["volume"].mean())
        dry_ratio = float(dry["volume"].median() / max(float(x.iloc[max(0, dry_start-20):dry_start]["volume"].median()), 1e-12))
        reexp_ratio = float(x.iloc[idx]["volume"] / max(dry_mean, 1e-12))
        # Require actual dry-up before the expansion.
        if dry_ratio <= low_volume_x and reexp_ratio >= second_expand_x:
            expansion_idx = idx
            expansion_ratio = reexp_ratio
            dry_mean_for_expansion = dry_mean
            break

    if expansion_idx is None:
        return None

    # Consolidation immediately before the expansion candle.
    zone_end = expansion_idx
    zone_start = zone_end - consolidation_bars
    if zone_start < 20:
        return None
    zone = x.iloc[zone_start:zone_end]
    if len(zone) < consolidation_bars:
        return None

    zone_high = float(zone["high"].max())
    zone_low = float(zone["low"].min())
    zone_mid = (zone_high + zone_low) / 2.0
    zone_range_pct = ((zone_high - zone_low) / zone_mid * 100.0) if zone_mid else 999.0
    if zone_range_pct > max_consolidation_pct:
        return None

    # Quantitative price slowdown: recent candle ranges are smaller than earlier ranges.
    sb = max(2, slowdown_bars // 2)
    slow_end = expansion_idx
    recent = x.iloc[max(zone_start, slow_end - sb):slow_end]
    earlier = x.iloc[max(zone_start, slow_end - slowdown_bars):max(zone_start, slow_end - sb)]
    if len(recent) < 2 or len(earlier) < 2:
        return None
    recent_range = float(recent["range_pct"].median())
    earlier_range = float(earlier["range_pct"].median())
    price_slowed = recent_range <= earlier_range * 0.80
    if not price_slowed:
        return None

    # First spike must occur before the dry-up/consolidation sequence.
    search_end = max(20, zone_start - low_bars)
    search_start = max(20, search_end - 100)
    prior = x.iloc[search_start:search_end]
    spike_candidates = prior[prior["vol_x"] >= first_spike_x]
    if spike_candidates.empty:
        return None

    first_spike_pos = int(spike_candidates.index[-1])
    first_spike_x_actual = float(x.loc[first_spike_pos, "vol_x"])

    # Breakout uses the expansion candle close versus the pre-expansion zone.
    candle = x.iloc[expansion_idx]
    close = float(candle["close"])
    buy_break = close > zone_high * (1.0 + breakout_buffer_pct / 100.0)
    sell_break = close < zone_low * (1.0 - breakout_buffer_pct / 100.0)

    if buy_break:
        signal = "BUY — VOLUME BREAKOUT"
    elif sell_break:
        signal = "SELL — VOLUME BREAKOUT"
    else:
        signal = "WATCH — VOLUME BUILDING"

    return {
        "signal": signal,
        "price": close,
        "first_spike_x": first_spike_x_actual,
        "reexpansion_x": expansion_ratio,
        "dry_mean": dry_mean_for_expansion,
        "zone_high": zone_high,
        "zone_low": zone_low,
        "zone_range_pct": zone_range_pct,
        "recent_range_pct": recent_range,
        "earlier_range_pct": earlier_range,
        "expansion_time": candle["time"],
        "first_spike_time": x.loc[first_spike_pos, "time"],
    }


def tradingview_url(pair):
    """Build a direct TradingView chart URL for the detected coin.

    CoinDCX futures pairs look like B-BTC_USDT. TradingView's direct
    chart URL is opened with the normalized BTCUSDT symbol. Binance is
    used as the primary TradingView market because it has broad USDT
    futures/spot coverage.
    """
    symbol = str(pair).replace("B-", "").replace("_USDT", "USDT")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}"


def coindcx_url(pair):
    return f"https://coindcx.com/futures/{pair}"


# -----------------------------
# Per coin scan
# -----------------------------

def scan_coin(row, volume_tf, settings):
    pair = row["Pair"]
    coin = row["Coin"]

    daily = get_klines(pair, "1D", limit=10)
    cont = two_day_continuation(daily)

    intraday = get_klines(pair, volume_tf, limit=220)
    vp = volume_pattern(intraday, **settings)

    if vp is None and cont["signal"] == "NONE":
        return None

    volume_signal = vp["signal"] if vp else "NONE"
    two_day_signal = cont["signal"]

    combined = "NONE"
    if two_day_signal == "BUY" and vp and volume_signal.startswith("BUY"):
        combined = "🟢 BUY — 2-DAY + VOLUME"
    elif two_day_signal == "SELL" and vp and volume_signal.startswith("SELL"):
        combined = "🔴 SELL — 2-DAY + VOLUME"
    elif two_day_signal != "NONE" and vp:
        combined = f"{two_day_signal} 2-DAY + {volume_signal}"
    elif two_day_signal != "NONE":
        combined = f"{two_day_signal} — 2-DAY CONTINUATION"
    elif vp:
        combined = volume_signal

    return {
        "Pair": pair,
        "Coin": coin,
        "Price": float(row["Last"]),
        "24h Volume": float(row["24h Volume"]),
        "2-Day": two_day_signal,
        "Volume Signal": volume_signal,
        "Combined": combined,
        "First Spike x": vp["first_spike_x"] if vp else np.nan,
        "Re-expansion x": vp["reexpansion_x"] if vp else np.nan,
        "Zone %": vp["zone_range_pct"] if vp else np.nan,
        "Zone High": vp["zone_high"] if vp else np.nan,
        "Zone Low": vp["zone_low"] if vp else np.nan,
        "Day 2 Vol / Day 1 Vol": cont.get("volume_ratio", np.nan),
    }


def run_scan(universe, volume_tf, settings):
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(scan_coin, row, volume_tf, settings): row["Coin"]
            for _, row in universe.iterrows()
        }
        for future in as_completed(futures):
            coin = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                errors.append((coin, str(e)))

    df = pd.DataFrame(results)
    if not df.empty:
        order = {
            "🟢 BUY — 2-DAY + VOLUME": 0,
            "🔴 SELL — 2-DAY + VOLUME": 1,
        }
        df["_order"] = df["Combined"].map(order).fillna(2)
        df = df.sort_values(["_order", "First Spike x"], ascending=[True, False]).drop(columns="_order")
    return df.reset_index(drop=True), errors


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="CoinDCX 2-Day + Volume Scanner",
    page_icon="📊",
    layout="wide",
)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "scan_signature" not in st.session_state:
    st.session_state.scan_signature = None
if "last_scan" not in st.session_state:
    st.session_state.last_scan = "Not scanned yet"
if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

st.title("📊 CoinDCX 2-Day Continuation + Volume Spike Scanner")
st.caption("Separate scanner • Daily 2-day continuation + selectable-timeframe volume pattern")

with st.sidebar:
    st.header("Scanner Settings")

    top_coins = st.selectbox("Coins to Scan", [50, 100, 150, 200, 250, 300], index=3)

    volume_tf = st.selectbox(
        "Volume Spike Timeframe",
        list(TIMEFRAME_OPTIONS.keys()),
        index=2,  # 5m
    )

    st.subheader("2-Day Continuation")
    st.caption("Daily candles are always used and only completed candles count.")
    enable_2day = st.toggle("Enable 2-Day Signal", True)

    st.subheader("Volume Pattern")
    first_spike_x = st.selectbox(
        "First Volume Spike",
        [2, 3, 5, 10, 20, 30, 50, 100],
        index=3,
        format_func=lambda x: f"{x}x",
    )
    second_expand_x = st.selectbox(
        "Second Volume Expansion",
        [1.2, 1.5, 2, 3, 5, 10],
        index=2,
        format_func=lambda x: f"{x}x vs dry-up",
    )
    low_volume_x = st.selectbox(
        "Low Volume Threshold",
        [0.40, 0.50, 0.60, 0.70, 0.80],
        index=2,
        format_func=lambda x: f"≤ {x:.2f}x baseline",
    )
    low_bars = st.slider("Low Volume Bars", 3, 10, 5)
    slowdown_bars = st.slider("Price Slowdown Bars", 4, 12, 6)
    consolidation_bars = st.slider("Consolidation Bars", 4, 12, 6)
    max_consolidation_pct = st.number_input("Max Consolidation Range %", 0.5, 10.0, 3.0, 0.5)
    breakout_buffer_pct = st.number_input("Breakout Buffer %", 0.0, 2.0, 0.10, 0.05)
    show_watch = st.toggle("Show WATCH setups", True)

    st.subheader("24h Filter")
    volume_filter = st.toggle("Minimum 24h Turnover", False)
    min_volume_m = st.number_input("Minimum 24h Volume ($M)", 0.0, 10000.0, 5.0, 1.0)

    st.subheader("Refresh")
    auto_refresh = st.toggle("Auto Refresh", False)
    refresh_minutes = st.selectbox("Refresh Every", [1, 2, 5, 10, 15], index=2)
    scan_now = st.button("🔍 SCAN NOW", use_container_width=True)

refresh_count = 0
if auto_refresh and st_autorefresh:
    refresh_count = st_autorefresh(
        interval=refresh_minutes * 60 * 1000,
        key="volume_scanner_refresh",
    )

signature = (
    int(top_coins), volume_tf, bool(enable_2day), float(first_spike_x),
    float(second_expand_x), float(low_volume_x), int(low_bars),
    int(slowdown_bars), int(consolidation_bars), float(max_consolidation_pct),
    float(breakout_buffer_pct), bool(show_watch), bool(volume_filter),
    float(min_volume_m), refresh_count,
)

if scan_now or st.session_state.scan_signature != signature:
    try:
        universe = get_symbols(int(top_coins))
        if volume_filter:
            universe = universe[universe["Turnover"] >= min_volume_m * 1_000_000].copy()

        settings = dict(
            first_spike_x=float(first_spike_x),
            second_expand_x=float(second_expand_x),
            low_volume_x=float(low_volume_x),
            low_bars=int(low_bars),
            slowdown_bars=int(slowdown_bars),
            consolidation_bars=int(consolidation_bars),
            max_consolidation_pct=float(max_consolidation_pct),
            breakout_buffer_pct=float(breakout_buffer_pct),
        )

        started = time.perf_counter()
        with st.spinner(f"Scanning {len(universe)} coins on {volume_tf}..."):
            df, errors = run_scan(universe, volume_tf, settings)
        elapsed = time.perf_counter() - started

        st.session_state.results = df
        st.session_state.scan_errors = errors
        st.session_state.scan_signature = signature
        st.session_state.last_scan = (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • {elapsed:.1f}s"
        )
    except Exception as e:
        st.error(f"Scanner error: {e}")

# -----------------------------
# Results
# -----------------------------

df = st.session_state.results.copy()

if df.empty:
    st.info("No matching setup found with the current settings. Try a lower First Volume Spike or wider consolidation range.")
else:
    combined_buy = int(df["Combined"].str.contains("🟢 BUY").sum())
    combined_sell = int(df["Combined"].str.contains("🔴 SELL").sum())
    watch_count = int(df["Volume Signal"].str.startswith("WATCH").sum())
    two_day_count = int((df["2-Day"] != "NONE").sum())

    a, b, c, d = st.columns(4)
    a.metric("🟢 BUY", combined_buy)
    b.metric("🔴 SELL", combined_sell)
    c.metric("🟡 WATCH", watch_count)
    d.metric("2-Day Signals", two_day_count)

    if st.session_state.scan_errors:
        st.warning(f"{len(st.session_state.scan_errors)} coins returned an error; successful results are shown.")

    # Combined first
    combined = df[df["Combined"].str.contains("2-DAY \\+ VOLUME", regex=True, na=False)]
    if not combined.empty:
        st.subheader("🔥 2-DAY + VOLUME CONFIRMATION")
        st.dataframe(
            combined[[
                "Coin", "Price", "Combined", "First Spike x",
                "Re-expansion x", "Zone %", "Day 2 Vol / Day 1 Vol"
            ]].style.format({
                "Price": "{:,.8g}",
                "First Spike x": "{:.1f}x",
                "Re-expansion x": "{:.1f}x",
                "Zone %": "{:.2f}%",
                "Day 2 Vol / Day 1 Vol": "{:.2f}x",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # 2-day continuation
    day2 = df[df["2-Day"] != "NONE"]
    if enable_2day and not day2.empty:
        st.subheader("📅 2-DAY CONTINUATION")
        st.dataframe(
            day2[[
                "Coin", "Price", "2-Day", "Day 2 Vol / Day 1 Vol",
                "Volume Signal", "24h Volume"
            ]].style.format({
                "Price": "{:,.8g}",
                "Day 2 Vol / Day 1 Vol": "{:.2f}x",
                "24h Volume": "{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Volume breakout
    vol_break = df[df["Volume Signal"].str.contains("VOLUME BREAKOUT", na=False)]
    if not vol_break.empty:
        st.subheader(f"🚀 VOLUME BREAKOUT — {volume_tf}")
        st.dataframe(
            vol_break[[
                "Coin", "Price", "Volume Signal", "First Spike x",
                "Re-expansion x", "Zone %", "Zone High", "Zone Low", "24h Volume"
            ]].style.format({
                "Price": "{:,.8g}",
                "First Spike x": "{:.1f}x",
                "Re-expansion x": "{:.1f}x",
                "Zone %": "{:.2f}%",
                "Zone High": "{:,.8g}",
                "Zone Low": "{:,.8g}",
                "24h Volume": "{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Watch setups
    if show_watch:
        watch = df[df["Volume Signal"].str.startswith("WATCH", na=False)]
        if not watch.empty:
            st.subheader(f"🟡 WATCH — VOLUME BUILDING ({volume_tf})")
            st.dataframe(
                watch[[
                    "Coin", "Price", "First Spike x", "Re-expansion x",
                    "Zone %", "Zone High", "Zone Low", "24h Volume"
                ]].style.format({
                    "Price": "{:,.8g}",
                    "First Spike x": "{:.1f}x",
                    "Re-expansion x": "{:.1f}x",
                    "Zone %": "{:.2f}%",
                    "Zone High": "{:,.8g}",
                    "Zone Low": "{:,.8g}",
                    "24h Volume": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("📈 Open Detected Coins")
    st.caption("TradingView opens the normalized coin directly. CoinDCX opens the exact futures pair.")

    for _, r in df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([1.7, 2.8, 1.5, 1.5, 1.5])
        c1.markdown(f"**{r['Coin']}**")
        c2.write(str(r["Combined"]))
        c3.metric("Price", f"{r['Price']:,.8g}")
        tv = tradingview_url(r["Pair"])
        dcx = coindcx_url(r["Pair"])
        c4.markdown(f'<a href="{tv}" target="_blank"><button style="width:100%;">📈 TradingView</button></a>', unsafe_allow_html=True)
        c5.markdown(f'<a href="{dcx}" target="_blank"><button style="width:100%;">↗ CoinDCX</button></a>', unsafe_allow_html=True)

    st.caption(
        f"Last scan: {st.session_state.last_scan} • Volume timeframe: {volume_tf} • "
        f"First spike: {first_spike_x}x • Re-expansion: {second_expand_x}x vs dry-up"
    )

# -----------------------------
# Optional chart for selected coin
# -----------------------------

st.divider()
st.subheader("📈 Inspect a Coin")

if not df.empty:
    selected = st.selectbox("Coin", df["Coin"].tolist())
    selected_pair = df.loc[df["Coin"] == selected, "Pair"].iloc[0]
    try:
        chart_df = get_klines(selected_pair, volume_tf, limit=120)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
        ))
        fig.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            title=f"{selected} • {volume_tf} completed candles",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.warning(f"Chart could not be loaded: {e}")

st.caption("Educational scanner only — signals are pattern detections, not guaranteed trade outcomes.")
