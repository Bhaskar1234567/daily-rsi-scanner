# CoinDCX Strategy Scanner
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

BASE = "https://api.coindcx.com"
PUBLIC_BASE = "https://public.coindcx.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CoinDCX-Strategy-Scanner/1.0"})
TIMEOUT = 12


@st.cache_data(ttl=300, show_spinner=False)
def get_symbols(limit=200):
    url = f"{BASE}/exchange/v1/derivatives/futures/data/active_instruments"
    r = SESSION.get(url, params={"margin_currency_short_name[]": "USDT"}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # CoinDCX may return the instruments directly as a list OR inside
    # a dictionary such as {"data": [...]}.
    if isinstance(data, dict):
        data = data.get("data") or data.get("instruments") or data.get("result") or []
    if not isinstance(data, list):
        raise RuntimeError("Unexpected active-instruments response from CoinDCX.")

    rows = []
    for x in data:
        if not isinstance(x, dict):
            continue
        pair = x.get("pair") or x.get("symbol")
        if not pair or not str(pair).endswith("_USDT"):
            continue
        last = float(x.get("last_price") or x.get("lastPrice") or 0)
        vol = float(x.get("volume_base") or x.get("volume") or 0)
        rows.append({
            "Pair": str(pair),
            "Coin": str(pair).replace("B-", "").replace("_USDT", ""),
            "24h Volume": vol,
            "Last": last,
            "Turnover": vol * last,
        })
    if not rows:
        raise RuntimeError("No active USDT futures instruments returned by CoinDCX.")
    return pd.DataFrame(rows).drop_duplicates("Pair").sort_values(
        "Turnover", ascending=False
    ).head(int(limit)).reset_index(drop=True)


def _parse_candles(data):
    if isinstance(data, dict):
        rows = data.get("data") or data.get("candles") or data.get("result") or []
    else:
        rows = data
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
            out.append([pd.to_datetime(ts, unit="s", utc=True),
                        float(o), float(h), float(l), float(c), float(v)])
        except Exception:
            continue
    if not out:
        raise RuntimeError("No usable candles returned.")
    return pd.DataFrame(
        out, columns=["time", "open", "high", "low", "close", "volume"]
    ).sort_values("time").drop_duplicates("time").reset_index(drop=True)


@st.cache_data(ttl=20, show_spinner=False)
def get_klines(pair, timeframe, limit=500):
    now = int(time.time())
    url = f"{PUBLIC_BASE}/market_data/candlesticks"

    if timeframe in ("2m", "3m"):
        bucket = 2 if timeframe == "2m" else 3
        one_min_limit = max(int(limit) * bucket + 20, 250)
        r = SESSION.get(url, params={
            "pair": pair, "from": now - one_min_limit * 60, "to": now,
            "resolution": "1", "pcode": "f"
        }, timeout=TIMEOUT)
        r.raise_for_status()
        raw = _parse_candles(r.json())
        raw["bucket"] = raw["time"].dt.floor(f"{bucket}min")
        current_bucket = pd.Timestamp.now(tz="UTC").floor(f"{bucket}min")
        raw = raw[raw["bucket"] < current_bucket]
        df = raw.groupby("bucket", sort=True).agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"),
            close=("close", "last"), volume=("volume", "sum"), count=("close", "size")
        ).reset_index().rename(columns={"bucket": "time"})
        return df[df["count"] == bucket].drop(columns="count").tail(int(limit)).reset_index(drop=True)

    if timeframe == "1D":
        r = SESSION.get(url, params={
            "pair": pair, "from": now - max(int(limit) + 10, 40) * 86400,
            "to": now, "resolution": "1D", "pcode": "f"
        }, timeout=TIMEOUT)
        r.raise_for_status()
        df = _parse_candles(r.json())
        today = pd.Timestamp.now(tz="UTC").floor("D")
        return df[df["time"] < today].tail(int(limit)).reset_index(drop=True)

    raise ValueError("Unsupported timeframe")


def rsi_wilder(close, period=34):
    close = pd.Series(close, dtype="float64")
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.mask((al == 0) & (ag > 0), 100).mask((ag == 0) & (al > 0), 0)


def classify_rsi(v, ob=70, near_ob=68, os=30, near_os=32):
    if v > ob:
        return "SHORT"
    if near_ob <= v <= ob:
        return "NEAR SHORT"
    if v < os:
        return "LONG"
    if os <= v <= near_os:
        return "NEAR LONG"
    return "NEUTRAL"


def previous_day_levels(daily):
    if len(daily) < 2:
        raise RuntimeError("Not enough completed daily candles.")
    row = daily.iloc[-1]
    return (row["high"] + row["close"]) / 2, (row["open"] + row["low"]) / 2, row["time"]


def transition_signal(df, up, down):
    last_zone = None
    found = None
    for _, row in df.iterrows():
        zone = "UP" if row["high"] >= up else "DOWN" if row["low"] <= down else None
        if zone is None:
            continue
        if last_zone == "DOWN" and zone == "UP":
            found = ("LONG", row)
        elif last_zone == "UP" and zone == "DOWN":
            found = ("SHORT", row)
        last_zone = zone
    return found


def scan_rsi(pair, timeframe, period, ob, near_ob, os, near_os, volume):
    df = get_klines(pair, timeframe, 180 if timeframe == "1D" else 300)
    if len(df) < period + 2:
        raise RuntimeError("Not enough completed candles for RSI.")
    value = float(rsi_wilder(df["close"], period).iloc[-1])
    return {
        "Pair": pair, "Coin": pair.replace("B-", "").replace("_USDT", ""),
        "Price": float(df["close"].iloc[-1]), "RSI": value,
        "Signal": classify_rsi(value, ob, near_ob, os, near_os),
        "24h Volume": float(volume)
    }


def scan_ohlc(pair, volume):
    m3 = get_klines(pair, "3m", 500)
    daily = get_klines(pair, "1D", 10)
    up, down, prev = previous_day_levels(daily)
    found = transition_signal(m3, up, down)
    return {
        "Pair": pair, "Coin": pair.replace("B-", "").replace("_USDT", ""),
        "Price": float(m3["close"].iloc[-1]),
        "Signal": found[0] if found else "WAIT",
        "UP Level": float(up), "DOWN Level": float(down),
        "Previous Day": prev.strftime("%Y-%m-%d"),
        "24h Volume": float(volume)
    }


def run_scan(strategy, universe, period, ob, near_ob, os, near_os):
    def worker(row):
        try:
            if strategy == "2M RSI":
                return scan_rsi(row["Pair"], "2m", period, ob, near_ob, os, near_os, row["24h Volume"])
            if strategy == "1D RSI":
                return scan_rsi(row["Pair"], "1D", period, ob, near_ob, os, near_os, row["24h Volume"])
            return scan_ohlc(row["Pair"], row["24h Volume"])
        except Exception as e:
            return {"__error__": True, "Pair": row["Pair"], "error": str(e)}

    results, errors = [], []
    workers = 6 if strategy != "1D RSI" else 10
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, row) for _, row in universe.iterrows()]
        for f in as_completed(futures):
            x = f.result()
            if x.get("__error__"):
                errors.append((x["Pair"], x["error"]))
            else:
                results.append(x)
    return pd.DataFrame(results), errors


@st.cache_data(ttl=20, show_spinner=False)
def load_chart(pair, strategy, period, ob, near_ob, os, near_os):
    if strategy in ("2M RSI", "1D RSI"):
        tf = "2m" if strategy == "2M RSI" else "1D"
        df = get_klines(pair, tf, 240 if tf == "2m" else 180)
        df["RSI"] = rsi_wilder(df["close"], period)
        return {"type": "rsi", "df": df}
    m3 = get_klines(pair, "3m", 360)
    daily = get_klines(pair, "1D", 10)
    up, down, prev = previous_day_levels(daily)
    return {"type": "ohlc", "df": m3, "up": float(up), "down": float(down), "prev": prev}


st.set_page_config(page_title="CoinDCX Strategy Scanner", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:2rem;}
[data-testid="stMetric"] {padding:10px;border-radius:10px;border:1px solid rgba(128,128,128,.18);}
a {text-decoration:none;}
</style>
""", unsafe_allow_html=True)

for key, default in [
    ("watchlist", []), ("selected_coin", None), ("results", pd.DataFrame()),
    ("scan_signature", None), ("scan_errors", []), ("last_scan_info", "Not scanned yet")
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("⚙️ Scanner")
    strategy = st.selectbox("Strategy", ["2M RSI", "1D RSI", "3M Previous-Day OHLC"])

    st.subheader("Coin Scan")
    top_coins = st.number_input("Top Coins", 10, 200, 200, 10)
    volume_filter = st.toggle("Volume Filter", False)
    min_volume_m = st.number_input("Minimum 24h Volume ($M)", 0.0, 10000.0, 10.0, 1.0) if volume_filter else 0.0

    if strategy in ("2M RSI", "1D RSI"):
        st.subheader("RSI Levels")
        rsi_period = st.number_input("RSI Period", 2, 200, 34, 1)
        overbought = st.number_input("SHORT above", 50.0, 100.0, 70.0, 1.0)
        near_overbought = st.number_input("Near SHORT from", 40.0, 99.0, 68.0, 1.0)
        oversold = st.number_input("LONG below", 0.0, 50.0, 30.0, 1.0)
        near_oversold = st.number_input("Near LONG to", 1.0, 60.0, 32.0, 1.0)
    else:
        rsi_period, overbought, near_overbought, oversold, near_oversold = 34, 70, 68, 30, 32

    st.subheader("Refresh")
    auto_refresh = st.toggle("Auto Refresh", False)
    refresh_minutes = st.selectbox("Refresh Every", [1, 2, 5, 10, 15], index=2)
    scan_now = st.button("🔄 SCAN NOW", use_container_width=True)

    st.divider()
    st.subheader("⭐ Watchlist")
    if st.session_state.watchlist:
        for coin in list(st.session_state.watchlist):
            c1, c2 = st.columns([3, 1])
            c1.write(coin)
            if c2.button("×", key=f"rm_{coin}"):
                st.session_state.watchlist.remove(coin)
                st.rerun()
    else:
        st.caption("Add coins with ⭐ from results.")

refresh_count = 0
if auto_refresh and st_autorefresh:
    refresh_count = st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

signature = (strategy, int(top_coins), volume_filter, float(min_volume_m),
             int(rsi_period), float(overbought), float(near_overbought),
             float(oversold), float(near_oversold), refresh_count)

if scan_now or st.session_state.scan_signature != signature:
    try:
        universe = get_symbols(int(top_coins))
        if volume_filter:
            universe = universe[universe["24h Volume"] >= min_volume_m * 1_000_000].copy()
        with st.spinner(f"Scanning {len(universe)} coins..."):
            results, errors = run_scan(strategy, universe, int(rsi_period),
                                      float(overbought), float(near_overbought),
                                      float(oversold), float(near_oversold))
        st.session_state.results = results
        st.session_state.scan_errors = errors
        st.session_state.scan_signature = signature
        st.session_state.last_scan_info = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception as e:
        st.error(f"Scanner error: {type(e).__name__}: {e}")

st.title("📈 CoinDCX Strategy Scanner")
st.caption("Simple live scanner • 📊 Show Chart below • ↗ CoinDCX opens a new tab")

df = st.session_state.results.copy()
if not df.empty:
    main = int(df["Signal"].isin(["LONG", "SHORT"]).sum())
    near = int(df["Signal"].astype(str).str.startswith("NEAR").sum())
    a, b, c, d = st.columns(4)
    a.metric("Coins Scanned", len(df))
    b.metric("Main Signals", main)
    c.metric("Near Signals", near)
    d.metric("Watchlist", len(st.session_state.watchlist))

    if st.session_state.scan_errors:
        st.warning(f"{len(st.session_state.scan_errors)} coins could not be loaded; successful results are shown.")

    def signal_rows(part, title, key_prefix, ohlc=False):
        st.markdown(f"### {title}")
        if part.empty:
            st.caption("No current signals.")
            return
        for _, row in part.iterrows():
            coin, pair = row["Coin"], row["Pair"]
            c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 1.4, 1])
            c1.write(f"**{coin}**")
            if ohlc:
                c2.write(f"UP **{row['UP Level']:,.6g}**")
                c3.write(f"DOWN **{row['DOWN Level']:,.6g}**")
                c4.write(f"Price **{row['Price']:,.6g}**")
            else:
                c2.write(f"RSI **{row['RSI']:.2f}**")
                c3.write(f"Price **{row['Price']:,.6g}**")
                c4.write(f"Vol **${row['24h Volume']/1e6:.1f}M**")
            if c5.button("📊 Chart", key=f"show_{key_prefix}_{coin}"):
                st.session_state.selected_coin = pair
                st.rerun()
            if c6.button("★" if coin in st.session_state.watchlist else "☆",
                         key=f"fav_{key_prefix}_{coin}"):
                if coin in st.session_state.watchlist:
                    st.session_state.watchlist.remove(coin)
                else:
                    st.session_state.watchlist.append(coin)
                st.rerun()
            st.markdown(
                f'<a href="https://coindcx.com/futures/{pair}" target="_blank">↗ Open CoinDCX Futures</a>',
                unsafe_allow_html=True
            )

    if strategy in ("2M RSI", "1D RSI"):
        for sig, title in [("LONG","🟢 LONG"), ("SHORT","🔴 SHORT"),
                           ("NEAR LONG","🟡 NEAR LONG"), ("NEAR SHORT","🟡 NEAR SHORT")]:
            signal_rows(df[df["Signal"] == sig], title, sig.replace(" ", "_"))
    else:
        signal_rows(df[df["Signal"] == "LONG"], "🟢 DOWN → UP — LONG", "long3m", True)
        signal_rows(df[df["Signal"] == "SHORT"], "🔴 UP → DOWN — SHORT", "short3m", True)

if st.session_state.selected_coin:
    pair = st.session_state.selected_coin
    coin = pair.replace("B-", "").replace("_USDT", "")
    st.divider()
    h1, h2 = st.columns([5, 2])
    h1.subheader(f"📊 {coin} — {strategy}")
    h2.markdown(
        f'<a href="https://coindcx.com/futures/{pair}" target="_blank">↗ Open CoinDCX in New Tab</a>',
        unsafe_allow_html=True
    )
    if st.button("✕ Close Chart"):
        st.session_state.selected_coin = None
        st.rerun()

    try:
        chart = load_chart(pair, strategy, int(rsi_period), float(overbought),
                           float(near_overbought), float(oversold), float(near_oversold))
        if chart["type"] == "rsi":
            x = chart["df"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.68, 0.32], vertical_spacing=0.04)
            fig.add_trace(go.Candlestick(x=x.time, open=x.open, high=x.high,
                                         low=x.low, close=x.close, name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x.time, y=x.RSI, mode="lines",
                                     name=f"RSI {rsi_period}"), row=2, col=1)
            for level, name in [(overbought,"SHORT"), (near_overbought,"Near SHORT"),
                                (near_oversold,"Near LONG"), (oversold,"LONG")]:
                fig.add_hline(y=level, line_dash="dash",
                              annotation_text=f"{name} {level:g}", row=2, col=1)
            fig.update_yaxes(range=[0,100], row=2, col=1)
            fig.update_layout(height=720, margin=dict(l=10,r=10,t=20,b=10),
                              xaxis_rangeslider_visible=False)
        else:
            x = chart["df"]
            fig = go.Figure(go.Candlestick(x=x.time, open=x.open, high=x.high,
                                           low=x.low, close=x.close, name="3M Price"))
            fig.add_hline(y=chart["up"], line_dash="dash",
                          annotation_text=f"UP {chart['up']:,.6g}")
            fig.add_hline(y=chart["down"], line_dash="dash",
                          annotation_text=f"DOWN {chart['down']:,.6g}")
            fig.update_layout(height=720, margin=dict(l=10,r=10,t=40,b=10),
                              xaxis_rangeslider_visible=False,
                              title=f"3M Previous-Day OHLC • Previous day {chart['prev'].strftime('%Y-%m-%d')}")
        st.plotly_chart(fig, use_container_width=True,
                        config={"displaylogo": False, "displayModeBar": True})
        st.caption("Use Plotly's expand button for full-screen chart view.")
    except Exception as e:
        st.error(f"Could not load chart: {e}")

st.divider()
st.subheader("⭐ Watchlist")
if not st.session_state.watchlist:
    st.caption("Add coins using ☆ beside any signal.")
else:
    for coin in st.session_state.watchlist:
        st.write(f"**{coin}**")

st.caption(f"Last scan: {st.session_state.last_scan_info} • Volume filter: {'ON' if volume_filter else 'OFF'}")
