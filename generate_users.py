import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime

SEED = 42
N_USERS = 10_000

# watch 생성 기간과 맞춰야 함
OBS_START = date(2025, 2, 1)
OBS_END   = date(2026, 2, 2)

OUT_PATH = "user.csv"

PLANS = ["basic", "standard", "premium"]
PLAN_P = [0.38, 0.44, 0.18]

GENDERS = ["M", "F"]
GENDER_P = [0.52, 0.48]

# 시나리오 비율(너희 기획서 기준)
# Non-Churn 25%, Gradual 30%, Early 15%, Dormant→Reactivated 15%, Pre-Churned 10%
RATIO_PRE_CHURNED = 0.10

# 관측기간 중 이탈 라벨을 "정답처럼 강제"하진 않되,
# 데이터에서 충분히 관측되게끔 user 단에서 일부는 churn_date를 부여해 둔다.
# (Gold에서 검증용 라벨로 사용)
RATIO_CHURN_DURING_WINDOW = 0.45  # (Pre 제외) 중 관측기간 중 이탈 라벨 비율 (권장 0.35~0.55)

def random_date(rng, start: date, end: date) -> date:
    days = (end - start).days
    return start + timedelta(days=int(rng.integers(0, days + 1)))

def main():
    rng = np.random.default_rng(SEED)

    user_id = np.arange(1, N_USERS + 1, dtype=int)
    age = np.clip(rng.normal(loc=34, scale=10, size=N_USERS).round().astype(int), 18, 70)
    gender = rng.choice(GENDERS, size=N_USERS, p=GENDER_P)
    plan = rng.choice(PLANS, size=N_USERS, p=PLAN_P)

    # join_date는 관측 시작보다 이전 가입자가 많게(현실감)
    join_start = date(2019, 1, 1)
    join_end = OBS_END
    join_dates = [random_date(rng, join_start, join_end) for _ in range(N_USERS)]

    # churn_date 생성 (빈 값은 유지/미라벨)
    churn_date = [""] * N_USERS

    # 1) Pre-Churned: 관측 시작 전 이탈 (10%)
    n_pre = int(round(N_USERS * RATIO_PRE_CHURNED))
    idx_all = np.arange(N_USERS)
    pre_idx = rng.choice(idx_all, size=n_pre, replace=False)

    for i in pre_idx:
        # 관측 시작 30~365일 전 사이
        cd = random_date(rng, OBS_START - timedelta(days=365), OBS_START - timedelta(days=30))
        # join_date는 churn_date보다 과거여야 자연스러움
        if join_dates[i] >= cd:
            join_dates[i] = random_date(rng, join_start, cd - timedelta(days=1))
        churn_date[i] = cd.isoformat()

    # 2) 관측기간 중 이탈 라벨: (Pre 제외한 나머지 중 일부)
    remaining = np.setdiff1d(idx_all, pre_idx)
    n_during = int(round(len(remaining) * RATIO_CHURN_DURING_WINDOW))
    during_idx = rng.choice(remaining, size=n_during, replace=False)

    for i in during_idx:
        # Early/Gradual/Dormant 등의 분포는 watch에서 자연스럽게 나오게 하되,
        # user churn_date는 "검증용 라벨"로만 존재하도록 날짜를 넓게 분포시킴.
        cd = random_date(rng, OBS_START + timedelta(days=10), OBS_END - timedelta(days=10))
        if join_dates[i] >= cd:
            join_dates[i] = random_date(rng, join_start, cd - timedelta(days=1))
        churn_date[i] = cd.isoformat()

    df = pd.DataFrame({
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "join_date": [d.isoformat() for d in join_dates],
        "plan": plan,
        "churn_date": churn_date,  # ✅ 추가 컬럼 (빈 값이면 관측기간 내 미이탈/미라벨)
    })

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # 간단 체크
    pre_cnt = (df["churn_date"] != "") & (pd.to_datetime(df["churn_date"]).dt.date < OBS_START)
    during_cnt = (df["churn_date"] != "") & (pd.to_datetime(df["churn_date"]).dt.date >= OBS_START) & (pd.to_datetime(df["churn_date"]).dt.date <= OBS_END)
    print(f"✅ created {OUT_PATH}: {len(df):,} users")
    print(f" - pre-churned: {pre_cnt.sum():,} ({pre_cnt.mean()*100:.1f}%)")
    print(f" - churn during window (labeled): {during_cnt.sum():,} ({during_cnt.mean()*100:.1f}%)")
    print(f" - unlabeled/retained: {(df['churn_date']== '').sum():,} ({(df['churn_date']== '').mean()*100:.1f}%)")

if __name__ == "__main__":
    main()