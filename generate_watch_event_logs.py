import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# 0) Config
# =========================================================
@dataclass
class Config:
    # 기간
    start_date: str = "2025-02-01"
    end_date: str = "2026-02-10"

    # 입력 파일
    user_csv_path: str = "user.csv"
    netflix_csv_path: str = "netflix.csv"

    # 출력
    output_dir: str = "watch_event_log_out"
    filename_pattern: str = "watch_event_log_{yyyymmdd}.csv"

    seed: int = 42

    devices: Tuple[str, ...] = ("mobile", "web", "tablet", "smart_tv")
    device_probs: Tuple[float, ...] = (0.55, 0.15, 0.10, 0.20)

    # 소수 유저 장르 편향
    genre_bias_user_ratio: float = 0.07
    genre_bias_strength: float = 4.0
    min_titles_per_genre: int = 50

    # 온보딩
    onboarding_days: int = 10
    onboarding_active_multiplier: float = 1.20   # 기존 1.25 → 살짝 완화
    onboarding_watch_multiplier: float = 1.10    # 기존 1.15 → 살짝 완화
    onboarding_time_multiplier: float = 1.15     # 기존 1.20 → 살짝 완화

    # 시간대 분포
    hour_weights: Tuple[Tuple[int, float], ...] = (
        (0, 0.15), (1, 0.08), (2, 0.05), (3, 0.03), (4, 0.03), (5, 0.05),
        (6, 0.08), (7, 0.12), (8, 0.18), (9, 0.25), (10, 0.30), (11, 0.35),
        (12, 0.45), (13, 0.55), (14, 0.60), (15, 0.65), (16, 0.75),
        (17, 0.90), (18, 1.00), (19, 1.05), (20, 1.05), (21, 0.95),
        (22, 0.75), (23, 0.45),
    )

    # 유저 타입 비율
    user_type_ratios: Dict[str, float] = None
    type_params: Dict[str, Dict] = None

    def __post_init__(self):
        # ✅ MA gives 너무 높게 나오는 원인: heavy+normal 비중/확률이 높음
        # → light 추가 + heavy 비중 축소 + 전반 p_active 하향
        if self.user_type_ratios is None:
            self.user_type_ratios = {
                "stable_heavy": 0.10,   # 25% → 10%
                "normal": 0.30,         # 35% → 30%
                "light": 0.32,          # ✅ 신규
                "weekend_binge": 0.12,  # 15% → 12%
                "chronic_churn": 0.10,  # 15% → 10%
                "sudden_churn": 0.06,   # 10% → 6%
            }

        if self.type_params is None:
            self.type_params = {
                "stable_heavy": {
                    "p_active_weekday": 0.70,   # 0.85 → 0.70
                    "p_active_weekend": 0.80,   # 0.92 → 0.80
                    "p_watch_given_active": 0.88,  # 0.92 → 0.88
                    "sessions_values": (1, 2, 3, 4),
                    "sessions_probs":  (0.40, 0.38, 0.18, 0.04),
                    "watch_mean_log": 3.25,
                    "watch_sigma_log": 0.55,
                },
                "normal": {
                    "p_active_weekday": 0.35,   # 0.55 → 0.35
                    "p_active_weekend": 0.45,   # 0.65 → 0.45
                    "p_watch_given_active": 0.72,  # 0.80 → 0.72
                    "sessions_values": (1, 2, 3),
                    "sessions_probs":  (0.75, 0.22, 0.03),
                    "watch_mean_log": 3.05,
                    "watch_sigma_log": 0.60,
                },
                "light": {
                    # ✅ “월 1~6일”을 만드는 핵심 타입
                    "p_active_weekday": 0.10,
                    "p_active_weekend": 0.16,
                    "p_watch_given_active": 0.60,
                    "sessions_values": (1, 2),
                    "sessions_probs":  (0.88, 0.12),
                    "watch_mean_log": 2.90,
                    "watch_sigma_log": 0.65,
                },
                "weekend_binge": {
                    "p_active_weekday": 0.05,   # 0.12 → 0.05
                    "p_active_weekend": 0.55,   # 0.75 → 0.55
                    "p_watch_given_active": 0.80,  # 0.88 → 0.80
                    "sessions_values": (2, 3, 4, 5),
                    "sessions_probs":  (0.20, 0.35, 0.30, 0.15),
                    "watch_mean_log": 3.25,
                    "watch_sigma_log": 0.65,
                },
                "chronic_churn": {
                    # 초반부터 normal과 똑같이 높으면 MAU가 계속 높음 → 약간 낮춤
                    "p_active_weekday": 0.30,
                    "p_active_weekend": 0.40,
                    "p_watch_given_active": 0.68,
                    "sessions_values": (1, 2, 3),
                    "sessions_probs":  (0.80, 0.18, 0.02),
                    "watch_mean_log": 3.00,
                    "watch_sigma_log": 0.65,
                    "decline_weeks": 4,
                    "decline_p_active_multiplier_end": 0.25,  # 0.35 → 0.25
                    "decline_watch_multiplier_end": 0.35,     # 0.45 → 0.35
                },
                "sudden_churn": {
                    "p_active_weekday": 0.30,
                    "p_active_weekend": 0.40,
                    "p_watch_given_active": 0.70,
                    "sessions_values": (1, 2, 3),
                    "sessions_probs":  (0.82, 0.16, 0.02),
                    "watch_mean_log": 3.00,
                    "watch_sigma_log": 0.60,
                },
            }


# =========================================================
# 1) Utils
# =========================================================
def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 0, None)
    s = p.sum()
    if s <= 0:
        raise ValueError("Probabilities sum to 0.")
    return p / s


def daterange(start: str, end: str) -> List[datetime]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e_dt = datetime.strptime(end, "%Y-%m-%d")
    if e_dt < s:
        raise ValueError(f"end_date < start_date: start={start}, end={end}")
    days = (e_dt - s).days
    return [s + timedelta(days=i) for i in range(days + 1)]


def is_weekend(day: datetime) -> bool:
    return day.weekday() >= 5


def sample_hour(rng: np.random.Generator, hour_weights: Tuple[Tuple[int, float], ...]) -> int:
    hours = np.array([h for h, _ in hour_weights], dtype=int)
    w = np.array([w for _, w in hour_weights], dtype=float)
    p = normalize_probs(w)
    return int(rng.choice(hours, p=p))


def clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


# =========================================================
# 2) Load masters
# =========================================================
def load_users(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.user_csv_path)

    required = {"user_id", "join_date"}
    if not required.issubset(df.columns):
        raise ValueError(f"user.csv must include columns: {sorted(required)}")

    df["user_id"] = df["user_id"].astype(int)
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce").dt.date
    if df["join_date"].isna().any():
        bad = df[df["join_date"].isna()].head(10)
        raise ValueError(f"join_date parse fail rows exist. sample:\n{bad}")

    # ✅ churn_date가 있으면 반영 (관측기간 중/전 이탈 유저 이벤트 생성 중단)
    if "churn_date" in df.columns:
        df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce").dt.date
    else:
        df["churn_date"] = pd.NaT

    return df


def load_netflix(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.netflix_csv_path)
    if "show_id" not in df.columns:
        raise ValueError("netflix.csv must include 'show_id' column.")
    df["show_id"] = df["show_id"].astype(str)
    if "listed_in" not in df.columns:
        df["listed_in"] = ""
    return df


def build_genre_maps(netflix_df: pd.DataFrame, min_titles_per_genre: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    all_show_ids = netflix_df["show_id"].astype(str).unique()

    genre_to_shows: Dict[str, List[str]] = {}
    for _, row in netflix_df.iterrows():
        sid = str(row["show_id"])
        listed = str(row.get("listed_in", "")).strip()
        if not listed:
            continue
        genres = [g.strip() for g in listed.split(",") if g.strip()]
        for g in genres:
            genre_to_shows.setdefault(g, []).append(sid)

    cleaned: Dict[str, np.ndarray] = {}
    for g, sids in genre_to_shows.items():
        uniq = np.unique(np.array(sids, dtype=str))
        if len(uniq) >= min_titles_per_genre:
            cleaned[g] = uniq

    return cleaned, all_show_ids


# =========================================================
# 3) Assign user types + churn schedules + genre biases
# =========================================================
def assign_user_types(cfg: Config, user_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    types = list(cfg.user_type_ratios.keys())
    probs = normalize_probs(np.array([cfg.user_type_ratios[t] for t in types], dtype=float))
    return rng.choice(types, size=len(user_ids), replace=True, p=probs)


def assign_sudden_churn_start(user_types: np.ndarray, days: List[datetime], rng: np.random.Generator) -> Dict[int, Optional[datetime]]:
    start_map: Dict[int, Optional[datetime]] = {}
    churn_candidates = days[30:-15] if len(days) >= 60 else days
    for idx, t in enumerate(user_types):
        start_map[idx] = rng.choice(churn_candidates) if t == "sudden_churn" else None
    return start_map


def assign_chronic_decline_start(user_types: np.ndarray, days: List[datetime], rng: np.random.Generator) -> Dict[int, Optional[datetime]]:
    start_map: Dict[int, Optional[datetime]] = {}
    safe_end_cut = 33
    candidates = days[10:-safe_end_cut] if len(days) > safe_end_cut + 10 else days
    for idx, t in enumerate(user_types):
        start_map[idx] = rng.choice(candidates) if t == "chronic_churn" else None
    return start_map


def assign_genre_bias_users(cfg: Config, user_ids: np.ndarray, genre_to_shows: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[int, Optional[str]]:
    bias_map: Dict[int, Optional[str]] = {i: None for i in range(len(user_ids))}
    if not genre_to_shows:
        return bias_map

    n_bias = int(round(len(user_ids) * cfg.genre_bias_user_ratio))
    n_bias = max(0, min(n_bias, len(user_ids)))
    biased_idxs = rng.choice(np.arange(len(user_ids)), size=n_bias, replace=False)

    genres = list(genre_to_shows.keys())
    counts = np.array([len(genre_to_shows[g]) for g in genres], dtype=float)
    p = normalize_probs(counts)

    for i in biased_idxs:
        bias_map[int(i)] = str(rng.choice(genres, p=p))

    return bias_map


# =========================================================
# 4) Core generation
# =========================================================
def get_decline_multipliers(cfg: Config, t: str, day: datetime, decline_start: Optional[datetime]) -> Tuple[float, float]:
    if t != "chronic_churn" or decline_start is None:
        return 1.0, 1.0

    params = cfg.type_params["chronic_churn"]
    weeks = int(params.get("decline_weeks", 4))
    end_active = float(params.get("decline_p_active_multiplier_end", 0.25))
    end_watch = float(params.get("decline_watch_multiplier_end", 0.35))

    delta_days = (day.date() - decline_start.date()).days
    if delta_days < 0:
        return 1.0, 1.0

    total = weeks * 7
    if delta_days >= total:
        return end_active, end_watch

    frac = delta_days / total
    active_mult = 1.0 + frac * (end_active - 1.0)
    watch_mult = 1.0 + frac * (end_watch - 1.0)
    return float(active_mult), float(watch_mult)


def sample_show_id(
    rng: np.random.Generator,
    all_show_ids: np.ndarray,
    genre_to_shows: Dict[str, np.ndarray],
    preferred_genre: Optional[str],
    bias_strength: float,
) -> str:
    if not preferred_genre or preferred_genre not in genre_to_shows:
        return str(rng.choice(all_show_ids))

    p_prefer = bias_strength / (bias_strength + 1.0)
    if rng.random() < p_prefer:
        return str(rng.choice(genre_to_shows[preferred_genre]))
    return str(rng.choice(all_show_ids))


def generate_day_events(
    cfg: Config,
    day: datetime,
    user_ids: np.ndarray,
    user_join_dates: np.ndarray,
    user_churn_dates: np.ndarray,  # ✅ 추가
    user_types: np.ndarray,
    sudden_churn_start_by_idx: Dict[int, Optional[datetime]],
    chronic_decline_start_by_idx: Dict[int, Optional[datetime]],
    preferred_genre_by_idx: Dict[int, Optional[str]],
    all_show_ids: np.ndarray,
    genre_to_shows: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    weekend = is_weekend(day)

    devs = np.array(cfg.devices, dtype=object)
    dev_p = normalize_probs(np.array(cfg.device_probs, dtype=float))

    for idx, uid in enumerate(user_ids):
        params = cfg.type_params[user_types[idx]]

        join_date = user_join_dates[idx]
        if day.date() < join_date:
            continue

        # ✅ user.csv churn_date 반영: 이탈 이후 이벤트 생성 X
        cd = user_churn_dates[idx]
        if pd.notna(cd) and day.date() > cd:
            continue
        if pd.notna(cd) and day.date() == cd:
            # 이탈 당일은 “아예 없거나 조금만” 나오게 싶으면 아래처럼
            if rng.random() < 0.6:
                continue

        # onboarding
        days_since_join = (day.date() - join_date).days
        in_onboarding = days_since_join < cfg.onboarding_days
        onboarding_active_mult = cfg.onboarding_active_multiplier if in_onboarding else 1.0
        onboarding_watch_mult = cfg.onboarding_watch_multiplier if in_onboarding else 1.0
        onboarding_time_mult = cfg.onboarding_time_multiplier if in_onboarding else 1.0

        # sudden churn: churn_start 이후 0
        churn_start = sudden_churn_start_by_idx.get(idx)
        if user_types[idx] == "sudden_churn" and churn_start is not None:
            if day.date() >= churn_start.date():
                continue

        p_active = float(params["p_active_weekend"] if weekend else params["p_active_weekday"])

        # chronic decline
        decline_start = chronic_decline_start_by_idx.get(idx)
        active_mult, watch_mult = get_decline_multipliers(cfg, user_types[idx], day, decline_start)

        p_active = float(np.clip(p_active * active_mult * onboarding_active_mult, 0.0, 0.98))
        if rng.random() >= p_active:
            continue

        # active but no watch 가능
        p_watch = float(np.clip(params["p_watch_given_active"] * onboarding_watch_mult, 0.0, 1.0))
        if rng.random() >= p_watch:
            continue

        sess_vals = np.array(params["sessions_values"], dtype=int)
        sess_probs = normalize_probs(np.array(params["sessions_probs"], dtype=float))
        n_sessions = int(rng.choice(sess_vals, p=sess_probs))

        pref_genre = preferred_genre_by_idx.get(idx)

        for _ in range(n_sessions):
            hour = sample_hour(rng, cfg.hour_weights)
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            event_ts = datetime(day.year, day.month, day.day, hour, minute, second).strftime("%Y-%m-%d %H:%M:%S")

            show_id = sample_show_id(
                rng=rng,
                all_show_ids=all_show_ids,
                genre_to_shows=genre_to_shows,
                preferred_genre=pref_genre,
                bias_strength=cfg.genre_bias_strength,
            )

            base = float(rng.lognormal(mean=float(params["watch_mean_log"]), sigma=float(params["watch_sigma_log"])))
            scaled = base * watch_mult * onboarding_time_mult
            session_time = clamp_int(scaled, 1, 240)

            device = str(rng.choice(devs, p=dev_p))

            rows.append({
                "event_ts": event_ts,
                "user_id": int(uid),
                "show_id": str(show_id),
                "session_time": int(session_time),
                "device": device,
            })

    df = pd.DataFrame(rows, columns=["event_ts", "user_id", "show_id", "session_time", "device"])
    if not df.empty:
        df = df.sort_values(["event_ts", "user_id"]).reset_index(drop=True)
    return df


def save_day_csv(cfg: Config, day: datetime, df: pd.DataFrame) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    yyyymmdd = day.strftime("%Y%m%d")
    out_path = os.path.join(cfg.output_dir, cfg.filename_pattern.format(yyyymmdd=yyyymmdd))
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# =========================================================
# 5) Main
# =========================================================
def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    print("=== CONFIG ===")
    for k, v in asdict(cfg).items():
        if k in ("type_params", "user_type_ratios"):
            print(f"{k}: (configured)")
        else:
            print(f"{k}: {v}")
    print()

    users_df = load_users(cfg)
    netflix_df = load_netflix(cfg)

    user_ids = users_df["user_id"].astype(int).to_numpy()
    user_join_dates = users_df["join_date"].to_numpy()
    user_churn_dates = users_df["churn_date"].to_numpy()  # ✅ 추가

    genre_to_shows, all_show_ids = build_genre_maps(netflix_df, cfg.min_titles_per_genre)

    print(f"Users: {len(user_ids):,}")
    print(f"Shows: {len(all_show_ids):,}")
    print(f"Genres (usable): {len(genre_to_shows):,}")

    days = daterange(cfg.start_date, cfg.end_date)
    print(f"Days: {len(days):,} ({cfg.start_date} ~ {cfg.end_date})")
    print()

    user_types = assign_user_types(cfg, user_ids, rng)
    sudden_churn_start_by_idx = assign_sudden_churn_start(user_types, days, rng)
    chronic_decline_start_by_idx = assign_chronic_decline_start(user_types, days, rng)

    preferred_genre_by_idx = assign_genre_bias_users(cfg, user_ids, genre_to_shows, rng)
    biased_count = sum(1 for v in preferred_genre_by_idx.values() if v is not None)
    print(f"Genre-biased users: {biased_count:,} ({biased_count/len(user_ids)*100:.2f}%)")
    print()

    total_rows = 0
    for i, day in enumerate(days, start=1):
        df = generate_day_events(
            cfg=cfg,
            day=day,
            user_ids=user_ids,
            user_join_dates=user_join_dates,
            user_churn_dates=user_churn_dates,
            user_types=user_types,
            sudden_churn_start_by_idx=sudden_churn_start_by_idx,
            chronic_decline_start_by_idx=chronic_decline_start_by_idx,
            preferred_genre_by_idx=preferred_genre_by_idx,
            all_show_ids=all_show_ids,
            genre_to_shows=genre_to_shows,
            rng=rng,
        )
        save_day_csv(cfg, day, df)
        total_rows += len(df)

        if i in (1, len(days)) or i % 10 == 0:
            print(f"[{i:4d}/{len(days)}] {day.strftime('%Y-%m-%d')} rows={len(df):,}")

    print()
    print(f"✅ Done. Total watch events: {total_rows:,}")
    print(f"📁 Output dir: {os.path.abspath(cfg.output_dir)}")
    print("Sample columns:", ["event_ts", "user_id", "show_id", "session_time", "device"])


if __name__ == "__main__":
    main()