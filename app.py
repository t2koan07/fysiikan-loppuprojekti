from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import requests
import certifi

from scipy.signal import butter, filtfilt, find_peaks, welch

import folium
from streamlit_folium import st_folium

# ============================
# GitHub raw settings
# ============================
GITHUB_USER = "t2koan07"
GITHUB_REPO = "fysiikan-loppuprojekti"
GITHUB_BRANCH = "main"

st.set_page_config(page_title="Fysiikan loppuprojekti", layout="wide")

# ============================
# Helpers
# ============================
def _read_git_remote_origin() -> Optional[str]:
    """Read remote origin URL from .git/config if available."""
    cfg = Path(".git") / "config"
    if not cfg.exists():
        return None
    try:
        txt = cfg.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    # Find [remote "origin"] section and its url
    m = re.search(r'\[remote "origin"\](.*?)(\n\[|$)', txt, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    um = re.search(r"url\s*=\s*(.+)", block)
    if not um:
        return None
    return um.group(1).strip()


def _parse_github_user_repo(remote_url: str) -> Optional[Tuple[str, str]]:
    """Parse GitHub user/repo from https or ssh remote URL."""
    # https://github.com/user/repo.git
    m = re.search(r"github\.com[:/](?P<user>[^/]+)/(?P<repo>[^/.]+)", remote_url)
    if not m:
        return None
    return m.group("user"), m.group("repo")


def _raw_base(user: str, repo: str, branch: str) -> str:
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/data/"

@st.cache_data(show_spinner=False)
def _read_csv_any(source: str) -> pd.DataFrame:
    """Read CSV from local path or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30, verify=certifi.where())
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text), comment="#")
    return pd.read_csv(source, comment="#")

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]

    # Partial match fallback
    for cand in candidates:
        cl = cand.lower()
        for k, orig in cols.items():
            if cl in k:
                return orig
    return None

def standardize_acc(df: pd.DataFrame) -> pd.DataFrame:
    t_col = _find_col(df, ["t", "time", "time (s)", "timestamp", "seconds"])
    x_col = _find_col(df, ["x", "ax", "linear acceleration x", "linear acceleration x (m/s^2)"])
    y_col = _find_col(df, ["y", "ay", "linear acceleration y", "linear acceleration y (m/s^2)"])
    z_col = _find_col(df, ["z", "az", "linear acceleration z", "linear acceleration z (m/s^2)"])

    if t_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError("Accelerometer.csv sarakkeita ei tunnistettu (t, x, y, z).")

    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df[t_col], errors="coerce"),
            "ax": pd.to_numeric(df[x_col], errors="coerce"),
            "ay": pd.to_numeric(df[y_col], errors="coerce"),
            "az": pd.to_numeric(df[z_col], errors="coerce"),
        }
    ).dropna()

    out = out.sort_values("t").reset_index(drop=True)
    out["a_mag"] = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2 + out["az"] ** 2)
    return out


def standardize_loc(df: pd.DataFrame) -> pd.DataFrame:
    t_col = _find_col(df, ["t", "time", "time (s)", "timestamp", "seconds"])
    lat_col = _find_col(df, ["lat", "latitude", "latitude (°)", "latitude (deg)"])
    lon_col = _find_col(df, ["lon", "lng", "longitude", "longitude (°)", "longitude (deg)"])

    if t_col is None or lat_col is None or lon_col is None:
        raise ValueError("Location.csv sarakkeita ei tunnistettu (t, lat, lon).")

    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df[t_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    ).dropna()

    out = out.sort_values("t").reset_index(drop=True)
    return out


def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    b, a = butter(order, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, x)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def distance_from_gps(loc: pd.DataFrame) -> float:
    if len(loc) < 2:
        return 0.0
    lat = loc["lat"].to_numpy()
    lon = loc["lon"].to_numpy()
    d = 0.0
    for i in range(1, len(loc)):
        d += haversine_m(float(lat[i - 1]), float(lon[i - 1]), float(lat[i]), float(lon[i]))
    return float(d)

def avg_speed_from_gps(loc: pd.DataFrame) -> float:
    if len(loc) < 2:
        return 0.0
    dt = float(loc["t"].iloc[-1] - loc["t"].iloc[0])
    if dt <= 0:
        return 0.0
    return distance_from_gps(loc) / dt

def _resolve_local_file(candidates: list[str]) -> Optional[Path]:
    data_dir = Path("data")
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None

def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load accelerometer and location data.
    Priority:
    1) Local data/ files (supports both the recommended names and common variants)
    2) GitHub raw URLs (requires user/repo/branch)
    """
    acc_local = _resolve_local_file(["Accelerometer.csv", "linear_accelerometer.csv", "accelerometer.csv"])
    loc_local = _resolve_local_file(["Location.csv", "location.csv"])

    if acc_local is not None and loc_local is not None:
        acc_raw = _read_csv_any(str(acc_local))
        loc_raw = _read_csv_any(str(loc_local))
        return acc_raw, loc_raw, "local"

    # Infer user/repo from git remote if not set
    user = (GITHUB_USER or "").strip()
    repo = (GITHUB_REPO or "").strip()
    branch = (GITHUB_BRANCH or "main").strip()

    if not user or not repo:
        origin = _read_git_remote_origin()
        if origin:
            parsed = _parse_github_user_repo(origin)
            if parsed:
                user, repo = parsed

    if not user or not repo:
        raise FileNotFoundError(
            "Dataa ei löytynyt paikallisesti (data/). Lisäksi GitHub USER/REPO puuttuu, "
            "joten dataa ei voida hakea raw.githubusercontent.com-osoitteesta."
        )

    base = _raw_base(user, repo, branch)

    acc_url_candidates = [base + "Accelerometer.csv", base + "linear_accelerometer.csv", base + "accelerometer.csv"]
    loc_url_candidates = [base + "Location.csv", base + "location.csv"]

    last_err = None
    acc_raw = None
    for u in acc_url_candidates:
        try:
            acc_raw = _read_csv_any(u)
            break
        except Exception as e:
            last_err = e

    if acc_raw is None:
        raise FileNotFoundError(f"Accelerometer-dataa ei saatu GitHubista. Viimeisin virhe: {last_err}")

    last_err = None
    loc_raw = None
    for u in loc_url_candidates:
        try:
            loc_raw = _read_csv_any(u)
            break
        except Exception as e:
            last_err = e

    if loc_raw is None:
        raise FileNotFoundError(f"Location-dataa ei saatu GitHubista. Viimeisin virhe: {last_err}")

    return acc_raw, loc_raw, "github"


# ============================
# UI
# ============================
st.title("Fysiikan loppuprojekti")

with st.sidebar:
    st.header("Asetukset")

    st.subheader("Data (GitHub-ajon tuki)")
    st.caption("URL-ajossa (streamlit run <url>) data luetaan raw.githubusercontent.com-osoitteesta.")
    st.caption(f"GitHub raw: {GITHUB_USER}/{GITHUB_REPO}@{GITHUB_BRANCH}")

    if not GITHUB_USER or not GITHUB_REPO:
        st.warning("GitHub USER/REPO puuttuu. URL-ajossa dataa ei voida hakea ilman niitä.")

try:
    acc_raw, loc_raw, source = load_dataframes()
except Exception as e:
    st.error(str(e))
    st.stop()

try:
    acc = standardize_acc(acc_raw)
    loc = standardize_loc(loc_raw)
except Exception as e:
    st.error(f"CSV-muotoa ei saatu tulkittua: {e}")
    st.stop()

# Time trimming (recommended because GPS lock may take time)
t_min = max(float(acc["t"].min()), float(loc["t"].min()))
t_max = min(float(acc["t"].max()), float(loc["t"].max()))
if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
    st.error("Aikajänne ei ole kelvollinen. Tarkista, että molemmissa CSV:issä on sama aikayksikkö (sekunteja).")
    st.stop()

with st.sidebar:
    st.subheader("Aikaväli analyysiin")
    t0, t1 = st.slider(
        "Valitse analysoitava aikaväli (s)",
        min_value=float(t_min),
        max_value=float(t_max),
        value=(float(t_min), float(t_max)),
        step=0.5,
    )

acc = acc[(acc["t"] >= t0) & (acc["t"] <= t1)].reset_index(drop=True)
loc = loc[(loc["t"] >= t0) & (loc["t"] <= t1)].reset_index(drop=True)

fs = estimate_fs(acc["t"].to_numpy())
duration_s = float(acc["t"].iloc[-1] - acc["t"].iloc[0]) if len(acc) >= 2 else 0.0

with st.sidebar:
    st.subheader("Kiihtyvyyskomponentti")
    component = st.selectbox(
        "Valitse analysoitava komponentti",
        options=["ax", "ay", "az", "a_mag"],
        index=2,
        help="Valitse komponentti, jossa askelrytmi erottuu selkeimmin. Tämä vaikuttaa suodatukseen ja askelmäärän arvioon.",
    )

    st.subheader("Suodatus (askelmäärä 1)")
    low = st.slider("Alarajataajuus (Hz)", 0.3, 2.0, 0.7, 0.1)
    high = st.slider("Ylärajataajuus (Hz)", 2.0, 10.0, 4.0, 0.5)
    peak_prom = st.slider("Peak prominence", 0.05, 3.0, 0.4, 0.05)

    st.subheader("Fourier (askelmäärä 2)")
    fmin = st.slider("Etsintäalue min (Hz)", 0.3, 2.0, 0.7, 0.1)
    fmax = st.slider("Etsintäalue max (Hz)", 2.0, 6.0, 3.5, 0.1)
    cycles_to_steps = st.selectbox("Syklit -> askeleet", options=[1, 2], index=0)

# ============================
# Calculations
# ============================
if fs <= 0 or len(acc) < 10:
    st.error("Kiihtyvyysdataa on liian vähän tai näytteenottotaajuutta ei voitu arvioida.")
    st.stop()

t = acc["t"].to_numpy()
x = acc[component].to_numpy()

# Remove DC offset (background) before filtering and PSD
x_detrended = x - float(np.mean(x))

# Bandpass filter for step counting
xf = butter_bandpass(x_detrended, fs, low, high, order=4)

# Step count method 1: peaks in filtered signal
max_step_hz = 4.0
min_dist = int(max(1, fs / max_step_hz))
peaks, _ = find_peaks(xf, distance=min_dist, prominence=peak_prom)
steps_filtered = int(len(peaks))

# Step count method 2: dominant frequency from PSD (Welch)
f, pxx = welch(x_detrended, fs=fs, nperseg=min(2048, len(x_detrended)))
mask = (f >= fmin) & (f <= fmax)
dominant_f = float(f[mask][np.argmax(pxx[mask])]) if np.any(mask) else 0.0
steps_fourier = int(round((dominant_f * duration_s) * cycles_to_steps)) if duration_s > 0 else 0

# GPS metrics
dist_m = distance_from_gps(loc)
avg_v = avg_speed_from_gps(loc)

# Step length (use filtered step count as primary)
step_len = (dist_m / steps_filtered) if steps_filtered > 0 else 0.0

# ============================
# Output: metrics
# ============================
st.caption(f"Data source: {source}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Kesto", f"{duration_s:.1f} s")
c2.metric("Näytteenottotaajuus", f"{fs:.1f} Hz")
c3.metric("Matka (GPS)", f"{dist_m:.0f} m")
c4.metric("Keskinopeus (GPS)", f"{avg_v:.2f} m/s")
c5.metric("Askelpituus", f"{step_len:.2f} m")

c6, c7, c8 = st.columns(3)
c6.metric("Askelmäärä (suodatettu)", f"{steps_filtered:d}")
c7.metric("Askelmäärä (Fourier)", f"{steps_fourier:d}")
c8.metric("Päätaajuus (Hz)", f"{dominant_f:.2f}")

# ============================
# Plot 1: filtered acceleration used for step counting
# ============================
st.subheader("Suodatettu kiihtyvyysdata (askelmäärä)")
fig1 = plt.figure()
plt.plot(t, xf, label="Suodatettu signaali")
if len(peaks) > 0:
    plt.plot(t[peaks], xf[peaks], "x", label="Askelpiikit")
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys (m/s²)")
plt.grid(True)
plt.legend()
st.pyplot(fig1, clear_figure=True)

# ============================
# Plot 2: PSD of chosen component
# ============================
st.subheader("Tehospektritiheys (PSD) valitusta komponentista")
fig2 = plt.figure()
plt.semilogy(f, pxx)
if dominant_f > 0:
    plt.axvline(dominant_f, linestyle="--")
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Tehospektritiheys")
plt.grid(True)
st.pyplot(fig2, clear_figure=True)

# ============================
# Plot 3: route on map
# ============================
st.subheader("Reitti kartalla")
if len(loc) >= 2:
    center = [float(loc["lat"].mean()), float(loc["lon"].mean())]
    m = folium.Map(location=center, zoom_start=15)

    points = list(zip(loc["lat"].to_numpy(), loc["lon"].to_numpy()))
    folium.PolyLine(points, weight=5).add_to(m)

    show_markers = st.checkbox("Näytä Start/End markerit", value=True)
    if show_markers:
        folium.Marker(points[0], tooltip="Start").add_to(m)
        folium.Marker(points[-1], tooltip="End").add_to(m)

    st_folium(m, width=900, height=500)
else:
    st.warning("GPS-dataa on liian vähän kartan piirtämiseen.")
