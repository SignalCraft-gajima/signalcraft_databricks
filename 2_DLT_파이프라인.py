import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType

# =========================================================
# 📌 SIGNALCRAFT OTT CHURN PREVENTION PIPELINE (DLT)
# =========================================================
# 목적:
#   1) 시청 이벤트(과거 Delta + 실시간 Capture)를 통합해 일별 시청시간을 만든다.
#   2) 유저 행동 스냅샷을 생성하고(T-1 확정본), 서비스/이탈 KPI 및 리텐션을 만든다.
#   3) 마케팅 전략 확정 테이블(dlt_gold_campaign_targets)을 생성한다.
#
# 레이어:
#   - Bronze: 원천(파일/캡처) 로드, JSON 파싱, 오류 격리
#   - Silver: 이벤트 통합, 일별 집계, Full Matrix(미접속=0 포함)
#   - Gold  : Snapshot / KPI / Retention / Campaign Targets
#
# 핵심 원칙:
#   - 운영 가정(T-1 스냅샷):
#       오늘 들어온 데이터는 완전하지 않을 수 있으므로
#       "데이터 기준 max(event_date) - 1"을 확정본으로 사용한다.
#   - Campaign Targets는 RAG 입력 SSOT:
#       strategy_code / priority_rank / send_flag 기준으로
#       메시지 생성(RAG)이 이뤄진다.
#       (RAG 결과 로그는 Gold를 늘리지 않기 위해 Silver로 저장 권장)
# =========================================================


# =========================================================
# 0) 공통 설정
# =========================================================
CATALOG = "signalcraft_databricks"
SCHEMA  = "default"

# 원천 데이터 소스 (Storage)
USER_SRC  = "abfss://signalcraft-data@signalcraftstorage.dfs.core.windows.net/user/"
CONT_SRC  = "abfss://signalcraft-data@signalcraftstorage.dfs.core.windows.net/contents/"

# 과거(1년치) 원천 로그 Delta 테이블
HISTORY_TABLE = f"{CATALOG}.{SCHEMA}.bronze_watch_history"

# Event Hubs Capture (AVRO) 소스 경로
WATCH_CAPTURE_SRC = "abfss://signalcraft-data@signalcraftstorage.dfs.core.windows.net/signalcraft-eventhub/http-events/"
PAYLOAD_COL = "Body"

# ✅ 모델 결과(SSOT) 테이블
MODEL_PRED_TABLE = f"{CATALOG}.{SCHEMA}.gold_churn_predictions"


# =========================================================
# 0-1) 공통 스키마/유틸 함수
# =========================================================

# Capture payload(JSON) 스키마: 일단 전부 String으로 받아서 안전하게 처리
PAYLOAD_JSON_SCHEMA = StructType([
    StructField("event_ts", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("show_id", StringType(), True),
    StructField("session_time", StringType(), True),
    StructField("device", StringType(), True),
])

# JSON payload에서 반드시 필요한 컬럼들
REQUIRED_COLS = ["event_ts", "user_id", "show_id", "session_time", "device"]


def parse_event_ts(colname="event_ts"):
    """
    문자열 timestamp를 최대한 안전하게 timestamp로 파싱한다.
    - 다양한 포맷을 순차적으로 시도
    - 실패 시 null (하위 로직에서 필터링)
    """
    return F.coalesce(
        F.expr(f"try_to_timestamp({colname}, 'yyyy-MM-dd HH:mm:ss.SSS')"),
        F.expr(f"try_to_timestamp({colname}, 'yyyy-MM-dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp({colname})"),
    )


def snapshot_cutoff_from_data(full_df):
    """
    📌 데이터 기준 T-1 스냅샷 확정 로직

    운영 환경 가정:
      - 오늘(최신 날짜) 데이터는 아직 불완전/부분일 수 있음
      - 따라서 snapshot_date = max(event_date) - 1 로 확정본을 만든다.

    예:
      max(event_date) = 2026-02-14
      → snapshot_date = 2026-02-13
    """
    max_date_df = full_df.agg(F.max("event_date").alias("max_event_date"))
    return max_date_df.select(F.date_sub(F.col("max_event_date"), 1).alias("snapshot_date"))


# =========================================================
# 1) BRONZE LAYER
# =========================================================
# 목적:
#   - 사용자/콘텐츠 마스터 원본 로드
#   - 과거 Delta(히스토리) 스트림 로드
#   - Event Hubs Capture(AVRO) 로드
#   - Capture의 JSON payload 파싱
#   - 파싱 실패/필수 컬럼 누락 레코드 격리(bad 테이블)
# =========================================================

@dlt.table(name="dlt_bronze_user", comment="(Bronze) 유저 마스터 원본")
def dlt_bronze_user():
    return spark.read.format("csv").option("header", "true").load(USER_SRC)


@dlt.table(name="dlt_bronze_netflix_master", comment="(Bronze) 넷플릭스 콘텐츠 마스터 원본")
def dlt_bronze_netflix_master():
    return spark.read.format("csv").option("header", "true").load(CONT_SRC)


@dlt.table(name="dlt_bronze_watch_history_stream", comment="(Bronze) 과거(Delta) 시청 로그를 스트리밍으로 읽기")
def dlt_bronze_watch_history_stream():
    return (spark.readStream.table(HISTORY_TABLE)
            .select("event_ts", "user_id", "show_id", "session_time", "device"))


@dlt.table(name="dlt_bronze_watch_capture_raw", comment="(Bronze) Event Hubs Capture AVRO raw (디버깅용)")
def dlt_bronze_watch_capture_raw():
    """
    Capture 원본을 그대로 적재한다.
    - file_path, ingest_ts를 붙여서 디버깅 가능하게 함
    """
    raw = (spark.readStream
           .format("cloudFiles")
           .option("cloudFiles.format", "avro")
           .option("cloudFiles.includeExistingFiles", "true")
           .load(WATCH_CAPTURE_SRC))
    return (raw
            .withColumn("file_path", F.col("_metadata.file_path"))
            .withColumn("ingest_ts", F.current_timestamp())
           )


@dlt.table(name="dlt_bronze_watch_event_log_bad", comment="(Bronze) 실시간 로그 파싱 실패/필수누락 격리 테이블")
def dlt_bronze_watch_event_log_bad():
    """
    JSON 파싱 실패 / payload 비정상 / 필수 컬럼 누락 레코드를 격리
    - 운영 시 데이터 품질 모니터링/원인 분석에 사용
    """
    raw = dlt.read_stream("dlt_bronze_watch_capture_raw")

    raw = raw.withColumn("payload_str", F.col(PAYLOAD_COL).cast("string"))
    parsed = raw.withColumn("j", F.from_json(F.col("payload_str"), PAYLOAD_JSON_SCHEMA, {"mode": "PERMISSIVE"}))

    too_short = F.col("payload_str").isNull() | (F.length(F.col("payload_str")) <= 2)
    parse_fail = F.col("j").isNull()

    cond_required = None
    for c in REQUIRED_COLS:
        expr = F.col(f"j.{c}").isNotNull()
        cond_required = expr if cond_required is None else (cond_required & expr)
    missing_required = ~cond_required

    return (parsed
        .filter(too_short | parse_fail | missing_required)
        .select(
            "payload_str",
            "file_path",
            "ingest_ts",
            F.when(too_short, F.lit("too_short_or_null"))
             .when(parse_fail, F.lit("json_parse_fail"))
             .otherwise(F.lit("missing_required")).alias("bad_reason")
        )
    )


@dlt.table(name="dlt_bronze_watch_event_log", comment="(Bronze) 실시간 시청 로그 (정상 payload만)")
def dlt_bronze_watch_event_log():
    """
    Capture raw에서 payload(JSON)를 파싱하여 정상 레코드만 추출
    """
    raw = dlt.read_stream("dlt_bronze_watch_capture_raw")

    raw = raw.filter(F.col(PAYLOAD_COL).isNotNull())
    with_payload = (raw
        .withColumn("payload_str", F.col(PAYLOAD_COL).cast("string"))
        .filter(F.length(F.col("payload_str")) > 2)
    )

    parsed = with_payload.withColumn(
        "j", F.from_json(F.col("payload_str"), PAYLOAD_JSON_SCHEMA, {"mode": "PERMISSIVE"})
    )
    parsed = parsed.filter(F.col("j").isNotNull())

    cond_required = None
    for c in REQUIRED_COLS:
        expr = F.col(f"j.{c}").isNotNull()
        cond_required = expr if cond_required is None else (cond_required & expr)
    parsed = parsed.filter(cond_required)

    return (parsed.select(
            F.col("j.event_ts").alias("event_ts"),
            F.col("j.user_id").alias("user_id"),
            F.col("j.show_id").alias("show_id"),
            F.col("j.session_time").alias("session_time"),
            F.col("j.device").alias("device"),
            "file_path",
            "ingest_ts"
        )
    )


# =========================================================
# 2) SILVER LAYER
# =========================================================
# 목적:
#   - 이벤트 레벨 데이터를 통합/정제한다.
#   - 유저×일자 단위 일별 시청시간을 만든다.
#   - Full Matrix(미접속=0)로 확장하여 inactivity 계산 기반을 만든다.
#
# 핵심:
#   - history(과거 Delta) + rt(실시간 Capture)를 union
#   - watermark로 지연 이벤트를 허용
#   - event_ts는 Asia/Seoul 기준 날짜(event_date)로 변환
# =========================================================

@dlt.table(name="dlt_silver_watch_events_all", comment="(Silver, Streaming) 과거(Delta)+실시간 이벤트 통합 (집계 전)")
def dlt_silver_watch_events_all():
    hist = dlt.read_stream("dlt_bronze_watch_history_stream")
    rt   = dlt.read_stream("dlt_bronze_watch_event_log")

    # 이벤트 레벨 통합
    u = hist.unionByName(
        rt.select("event_ts", "user_id", "show_id", "session_time", "device"),
        allowMissingColumns=True
    )

    # 타입 정제 + timestamp 파싱 + 날짜 파생
    return (u
        .withColumn("event_ts_ts", parse_event_ts("event_ts"))
        .filter(F.col("event_ts_ts").isNotNull())
        .withColumn("user_id", F.col("user_id").cast("int"))
        .withColumn("session_time", F.col("session_time").cast("int"))
        .filter((F.col("session_time") >= 1) & (F.col("session_time") <= 1440))
        .withWatermark("event_ts_ts", "400 days")
        .withColumn("event_date", F.to_date(F.from_utc_timestamp(F.col("event_ts_ts"), "Asia/Seoul")))
    )


@dlt.table(name="dlt_silver_daily_watch_time_rt", comment="(Silver, Streaming) 실시간(apptesting)만 일별 시청시간")
def dlt_silver_daily_watch_time_rt():
    """
    (옵션) 실시간 데이터만 따로 보고 싶을 때 사용
    - 메인 집계는 daily_watch_time(과거+실시간 통합)을 사용
    """
    df = dlt.read_stream("dlt_bronze_watch_event_log")
    df = (df
          .withColumn("event_ts_ts", parse_event_ts("event_ts"))
          .filter(F.col("event_ts_ts").isNotNull())
          .withWatermark("event_ts_ts", "2 days")
          .withColumn("event_date", F.to_date(F.from_utc_timestamp(F.col("event_ts_ts"), "Asia/Seoul")))
          .withColumn("user_id", F.col("user_id").cast("int"))
          .withColumn("session_time", F.col("session_time").cast("int"))
          .filter((F.col("session_time") >= 1) & (F.col("session_time") <= 1440))
    )
    return (df.groupBy("event_date", "user_id")
              .agg(F.sum("session_time").alias("daily_watch_time"))
              .withColumn("is_active", F.lit(1))
    )


@dlt.table(name="dlt_silver_daily_watch_time", comment="(Silver, Streaming) 과거+실시간 통합 일별/유저별 시청시간")
def dlt_silver_daily_watch_time():
    """
    이벤트 통합 테이블(dlt_silver_watch_events_all)을 일별 집계
    """
    events = dlt.read_stream("dlt_silver_watch_events_all")
    return (events
        .groupBy("event_date", "user_id")
        .agg(F.sum("session_time").alias("daily_watch_time"))
        .withColumn("is_active", F.lit(1))
    )


@dlt.table(name="dlt_silver_daily_watch_time_full", comment="(Silver) 미접속 날짜(0분) 포함 Full Matrix")
def dlt_silver_daily_watch_time_full():
    """
    Full Matrix를 만드는 이유:
      - 미접속 일자를 0분/0활성으로 포함해야
        days_since_last_login, inactivity_index 같은 지표가 정확해짐.
      - 스냅샷/리텐션/전략매핑의 기반이 됨.
    """
    watch = dlt.read("dlt_silver_daily_watch_time")
    users = dlt.read("dlt_bronze_user").select(
        F.col("user_id").cast("int").alias("user_id"),
        F.to_date("join_date").alias("join_date")
    )

    bounds = watch.agg(F.min("event_date").alias("start"), F.max("event_date").alias("end"))
    dates = bounds.select(
        F.explode(F.sequence(F.col("start"), F.col("end"), F.expr("interval 1 day"))).alias("event_date")
    )

    return (users.crossJoin(dates)
            .filter(F.col("event_date") >= F.col("join_date"))
            .join(watch, on=["event_date", "user_id"], how="left")
            .fillna({"daily_watch_time": 0, "is_active": 0})
    )


# =========================================================
# 3) GOLD LAYER - Snapshot / KPI / Retention
# =========================================================
# 목적:
#   - dlt_gold_user_behavior_snapshot: 유저별 일자 스냅샷(행동/위험)
#   - dlt_gold_service_kpi           : 서비스 KPI(DAU/WAU/MAU 등)
#   - dlt_gold_churn_risk_kpi        : 이탈 위험 KPI(재고/유입/회복)
#   - dlt_gold_retention             : 코호트 리텐션
#
# 공통:
#   - snapshot_cutoff_from_data()로 T-1 확정본만 남긴다.
# =========================================================

@dlt.table(
    name="dlt_gold_user_behavior_snapshot",
    comment="""
    (Gold) 유저 행태 스냅샷 - 데이터 기준 T-1 확정본

    주요 컬럼:
      - daily_watch_time_min / watch_time_7d_min / watch_time_30d_min
      - active_days_7 / active_days_30
      - days_since_last_login / segment
      - churn_reason / churn_risk_level
      - (추가) observation_days / frequency_active_days (전략 매핑용)

    ⚠ 이 테이블은 KPI/Retention/Campaign Targets의 기반 테이블입니다.
    """
)
def dlt_gold_user_behavior_snapshot():
    df = dlt.read("dlt_silver_daily_watch_time_full")
    cutoff = snapshot_cutoff_from_data(df)

    w7    = Window.partitionBy("user_id").orderBy("event_date").rowsBetween(-6, 0)
    w30   = Window.partitionBy("user_id").orderBy("event_date").rowsBetween(-29, 0)
    w_all = Window.partitionBy("user_id").orderBy("event_date").rowsBetween(Window.unboundedPreceding, 0)

    # 1) 기본 행동 지표
    base = (df
        .withColumn("watch_time_7d_min",  F.sum("daily_watch_time").over(w7))
        .withColumn("watch_time_30d_min", F.sum("daily_watch_time").over(w30))
        .withColumn("active_days_7",      F.sum("is_active").over(w7).cast("int"))
        .withColumn("active_days_30",     F.sum("is_active").over(w30).cast("int"))
        .withColumn("last_login", F.max(F.when(F.col("is_active") == 1, F.col("event_date"))).over(w_all))
        .withColumn("days_since_last_login", F.datediff("event_date", "last_login"))
        .withColumn(
            "segment",
            F.when(F.col("active_days_30") >= 20, F.lit("Heavy"))
             .when(F.col("active_days_30") >= 5,  F.lit("Mid"))
             .otherwise(F.lit("Light"))
        )
    )

    # 2) 전략/이탈 판단을 위한 보조 지표
    #    - observation_days: 가입일 기준 누적 관측일수
    #    - frequency_active_days: 가입 이후 누적 활성일수
    base = (base
        .withColumn("observation_days", F.datediff(F.col("event_date"), F.col("join_date")) + F.lit(1))
        .withColumn("frequency_active_days", F.sum("is_active").over(w_all).cast("int"))
        .withColumn(
            "mivt",
            F.when(F.col("frequency_active_days") > 0,
                   F.col("observation_days") / F.col("frequency_active_days").cast("double"))
             .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "inactivity_index",
            F.when((F.col("mivt").isNotNull()) & (F.col("mivt") > 0),
                   F.col("days_since_last_login").cast("double") / F.col("mivt"))
             .otherwise(F.lit(None).cast("double"))
        )
    )

    # 3) churn_reason(스냅샷 rule) & churn_risk_level
    #    - 모델의 churn_reason과 별개로, 스냅샷 관점 원인을 분류(리포팅용)
    ONBOARD_INACT_DAYS = 7
    SILENT_DECAY_MODE = "A"  # A/B 중 선택

    prechurned_cond = ((F.col("frequency_active_days") == 0) & (F.col("observation_days") >= 60))
    data_gap_cond = ((F.col("frequency_active_days").isin([1, 2])) & (F.col("observation_days") >= 30))
    onboarding_fail_cond = (
        (F.col("observation_days") <= 30) &
        (F.col("frequency_active_days") <= 2) &
        (F.col("days_since_last_login") >= ONBOARD_INACT_DAYS)
    )

    silent_decay_A = ((F.col("active_days_30") >= 6) & (F.col("active_days_7") <= 1))
    silent_decay_B = (
        (F.col("watch_time_30d_min") >= 120) &
        (F.col("watch_time_7d_min") <= (F.col("watch_time_30d_min") * F.lit(0.10)))
    )
    silent_decay_cond = F.when(F.lit(SILENT_DECAY_MODE) == F.lit("B"), silent_decay_B).otherwise(silent_decay_A)

    base = (base
        .withColumn(
            "churn_reason",
            F.when(prechurned_cond, F.lit("prechurned"))
             .when(data_gap_cond, F.lit("data_gap"))
             .when(onboarding_fail_cond, F.lit("onboarding_fail"))
             .when(silent_decay_cond, F.lit("silent_decay"))
             .otherwise(F.lit("normal"))
        )
    )

    base = (base
        .withColumn(
            "churn_risk_level",
            F.when(F.col("inactivity_index").isNull(), F.lit("Active"))
             .when(F.col("inactivity_index") < 1.0,    F.lit("Active"))
             .when(F.col("inactivity_index") < 2.0,    F.lit("Soft Churn"))
             .when(F.col("inactivity_index") < 5.0,    F.lit("Dormant"))
             .otherwise(F.lit("Churned"))
        )
    )

    # 4) 상태 보정(단순 rule)
    SOFT_DAYS = 14
    CHURN_DAYS_PRE = 60
    dsll = F.coalesce(F.col("days_since_last_login"), F.col("observation_days"))

    base = (base
        .withColumn(
            "churn_risk_level",
            F.when((F.col("churn_reason") == "prechurned") & (dsll >= CHURN_DAYS_PRE), F.lit("Churned"))
             .when((F.col("churn_reason") == "prechurned") & (dsll >= SOFT_DAYS),      F.lit("Dormant"))
             .when((F.col("churn_reason") == "prechurned") & (dsll <  SOFT_DAYS),      F.lit("Soft Churn"))
             .otherwise(F.col("churn_risk_level"))
        )
    )

    # 5) T-1 컷오프 적용 + 컬럼 정리
    snap = (base.crossJoin(cutoff)
            .filter(F.col("event_date") <= F.col("snapshot_date"))
            .select(
                "event_date", "user_id", "is_active",
                F.col("daily_watch_time").alias("daily_watch_time_min"),
                "watch_time_7d_min", "watch_time_30d_min",
                "active_days_7", "active_days_30",
                "days_since_last_login", "segment", "churn_risk_level",
                "churn_reason",
                "observation_days",
                "frequency_active_days"
            )
    )

    # (선택) probability_band를 붙여 Looker에서 조인 줄이기
    pred_band = (spark.table(MODEL_PRED_TABLE)
                .select(
                    F.col("event_date").alias("pred_event_date"),
                    F.col("user_id").alias("pred_user_id"),
                    F.col("probability_band").alias("probability_band")
                ))

    snap = (snap.alias("s")
            .join(pred_band.alias("p"),
                  (F.col("s.event_date") == F.col("p.pred_event_date")) &
                  (F.col("s.user_id") == F.col("p.pred_user_id")),
                  "left")
            .select("s.*", "p.probability_band")
    )

    return snap


@dlt.table(name="dlt_gold_service_kpi", comment="(Gold) 서비스 KPI - 데이터 기준 T-1 확정본")
def dlt_gold_service_kpi():
    full = dlt.read("dlt_silver_daily_watch_time_full")
    cutoff = snapshot_cutoff_from_data(full)

    daily_base = full.groupBy("event_date").agg(
        F.sum("is_active").alias("dau"),
        F.sum("daily_watch_time").alias("total_watch_time_min"),
        F.count("user_id").alias("total_users")
    )

    dates = full.select("event_date").distinct()
    active_logs = full.filter(F.col("is_active") == 1).select("event_date", "user_id")

    wau = (dates.alias("d").join(active_logs.alias("a"),
            (F.col("a.event_date") > F.date_sub(F.col("d.event_date"), 7)) &
            (F.col("a.event_date") <= F.col("d.event_date")))
        .groupBy("d.event_date").agg(F.countDistinct("user_id").alias("wau"))
    )

    mau = (dates.alias("d").join(active_logs.alias("a"),
            (F.col("a.event_date") > F.date_sub(F.col("d.event_date"), 30)) &
            (F.col("a.event_date") <= F.col("d.event_date")))
        .groupBy("d.event_date").agg(F.countDistinct("user_id").alias("mau"))
    )

    result = (daily_base.join(wau, "event_date")
        .join(mau, "event_date")
        .withColumn(
            "avg_watch_min",
            F.round(F.col("total_watch_time_min") / F.when(F.col("dau") == 0, F.lit(None)).otherwise(F.col("dau")), 2)
        )
        .withColumn(
            "active_ratio",
            F.round(F.col("dau") / F.when(F.col("total_users") == 0, F.lit(None)).otherwise(F.col("total_users")), 4)
        )
        .select("event_date", "dau", "wau", "mau", "total_watch_time_min", "avg_watch_min", "active_ratio")
    )

    return (result.crossJoin(cutoff)
            .filter(F.col("event_date") <= F.col("snapshot_date"))
    )


@dlt.table(name="dlt_gold_churn_risk_kpi", comment="(Gold) 이탈 위험 KPI - 데이터 기준 T-1 확정본")
def dlt_gold_churn_risk_kpi():
    snap = dlt.read("dlt_gold_user_behavior_snapshot").select("event_date", "user_id", "churn_risk_level")

    w_user = Window.partitionBy("user_id").orderBy("event_date")
    s = snap.withColumn("prev_churn_risk_level", F.lag("churn_risk_level", 1).over(w_user))

    is_risk_today = F.col("churn_risk_level").isin(["Dormant", "Churned"])
    is_risk_prev  = F.col("prev_churn_risk_level").isin(["Dormant", "Churned"])

    stock = s.groupBy("event_date").agg(
        F.countDistinct(F.when(F.col("churn_risk_level") == "Active", F.col("user_id"))).alias("active_cnt"),
        F.countDistinct(F.when(F.col("churn_risk_level") == "Soft Churn", F.col("user_id"))).alias("soft_churn_cnt"),
        F.countDistinct(F.when(F.col("churn_risk_level") == "Dormant", F.col("user_id"))).alias("dormant_cnt"),
        F.countDistinct(F.when(F.col("churn_risk_level") == "Churned", F.col("user_id"))).alias("churned_cnt"),
    )

    flow = s.groupBy("event_date").agg(
        F.countDistinct(F.when(is_risk_today, F.col("user_id"))).alias("at_risk_user_cnt"),
        F.countDistinct(F.when(is_risk_today & (~is_risk_prev | F.col("prev_churn_risk_level").isNull()), F.col("user_id"))).alias("at_risk_new_cnt"),
        F.countDistinct(F.when(is_risk_prev & (~is_risk_today), F.col("user_id"))).alias("at_risk_recovered_cnt"),
        F.countDistinct(F.when((F.col("churn_risk_level") == "Churned") & (F.col("prev_churn_risk_level").isin(["Dormant", "Soft Churn"])), F.col("user_id"))).alias("new_churned_cnt")
    )

    return (stock.join(flow, "event_date")
            .select("event_date","at_risk_user_cnt","at_risk_new_cnt","at_risk_recovered_cnt","new_churned_cnt",
                    "active_cnt","soft_churn_cnt","dormant_cnt","churned_cnt")
    )


@dlt.table(name="dlt_gold_retention", comment="(Gold) 리텐션 - 데이터 기준 T-1 확정본")
def dlt_gold_retention():
    snap = dlt.read("dlt_gold_user_behavior_snapshot").select("event_date","user_id","is_active","segment")

    first_seen = (snap.groupBy("user_id")
                  .agg(F.min("event_date").alias("first_event_date"))
                  .withColumn("cohort_month", F.date_trunc("MONTH", F.col("first_event_date"))))

    monthly_active = (snap
        .filter(F.col("is_active") == 1)
        .select("user_id", F.date_trunc("MONTH", F.col("event_date")).alias("event_month"))
        .distinct()
    )

    ret = (monthly_active.join(first_seen.select("user_id","cohort_month"), "user_id", "inner")
           .withColumn("months_since_join", F.months_between(F.col("event_month"), F.col("cohort_month")).cast("int"))
           .withColumn("is_retained", F.lit(1))
           .select("user_id","cohort_month","event_month","months_since_join","is_retained"))

    cohort_size = first_seen.groupBy("cohort_month").agg(F.countDistinct("user_id").alias("cohort_size"))

    seg_month = (snap
        .withColumn("event_month", F.date_trunc("MONTH", F.col("event_date")))
        .groupBy("user_id","event_month")
        .agg(F.max_by(F.col("segment"), F.col("event_date")).alias("segment_current"))
    )

    return (ret.join(cohort_size, "cohort_month", "left")
            .join(seg_month, ["user_id","event_month"], "left")
            .select("user_id","cohort_month","event_month","months_since_join","cohort_size","is_retained","segment_current")
    )