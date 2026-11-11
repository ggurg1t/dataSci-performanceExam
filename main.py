import math
import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

APP_TITLE = "분자 구조에 따른 수용해도 측정기"
APP_ICON = "🧪"
DATA_URL = (
    "https://raw.githubusercontent.com/dataprofessor/data/master/"
    "delaney_solubility_with_descriptors.csv"
)
TARGET = "logS"

# 열 설명 (Delaney / ESOL 데이터셋)
COLUMN_INFO = {
    "MolLogP": ("입력(특성)", "옥탄올-물 분배계수(logP). 클수록 물을 싫어하는(소수성) 분자"),
    "MolWt": ("입력(특성)", "분자량 (g/mol)"),
    "NumRotatableBonds": ("입력(특성)", "회전 가능한 결합의 수. 분자의 유연성"),
    "AromaticProportion": ("입력(특성)", "전체 무거운 원자 중 방향족 원자의 비율 (0~1)"),
    "logS": ("목표(정답)", "수용해도의 로그값 log10(mol/L). 클수록 물에 잘 녹음"),
}

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


@st.cache_resource(show_spinner="모델을 학습하는 중...")
def train_model(df: pd.DataFrame):
    features = [c for c in df.columns if c != TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[TARGET], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "model": model,
        "features": features,
        "y_train": y_train,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "r2_train": r2_score(y_train, pred_train),
        "r2_test": r2_score(y_test, pred_test),
        "mse_train": mean_squared_error(y_train, pred_train),
        "mse_test": mean_squared_error(y_test, pred_test),
    }


# ───────────────────────── 페이지 1: 데이터 정보 ─────────────────────────
def page_data_info(df: pd.DataFrame):
    st.header("📊 데이터베이스 정보")
    st.caption(
        "Delaney(ESOL) 수용해도 데이터셋 · 유기화합물의 분자 구조 기술자(descriptor)와 "
        "실측 수용해도(logS)가 담겨 있습니다."
    )
    st.markdown(f"**출처:** [{DATA_URL}]({DATA_URL})")

    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    n_dup = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("유기화합물 수 (행)", f"{n_rows:,}")
    c2.metric("열 수", f"{n_cols}")
    c3.metric("결측치 총 개수", f"{n_missing:,}")
    c4.metric("중복 행", f"{n_dup:,}")

    if n_missing == 0:
        st.success("모든 열에 결측치가 없습니다.")
    else:
        st.warning(f"결측치가 {n_missing:,}개 있습니다. 아래 표에서 열별 개수를 확인하세요.")

    st.subheader("열 구성과 종류")
    summary = pd.DataFrame(
        {
            "열 이름": df.columns,
            "자료형": [str(t) for t in df.dtypes],
            "역할": [COLUMN_INFO.get(c, ("-", "-"))[0] for c in df.columns],
            "설명": [COLUMN_INFO.get(c, ("-", "-"))[1] for c in df.columns],
            "값이 있는 수": df.notna().sum().values,
            "결측치 수": df.isna().sum().values,
            "고유값 수": df.nunique().values,
        }
    )
    st.dataframe(summary, hide_index=True)

    st.subheader("기초 통계량")
    st.dataframe(df.describe().T.round(3))

    st.subheader("데이터 미리보기")
    n_show = st.slider("보여줄 행 수", 5, 50, 10)
    st.dataframe(df.head(n_show))

    st.subheader("열별 분포")
    col = st.selectbox("열 선택", df.columns, index=list(df.columns).index(TARGET))
    fig = px.histogram(df, x=col, nbins=40, title=f"{col} 분포")
    fig.update_layout(yaxis_title="화합물 수", bargap=0.05)
    st.plotly_chart(fig)

    st.subheader("열 사이의 상관관계")
    corr = df.corr(numeric_only=True).round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="상관계수 (피어슨)",
    )
    st.plotly_chart(fig)

    st.subheader("특성과 logS의 관계")
    feat = st.selectbox(
        "x축 특성", [c for c in df.columns if c != TARGET], key="scatter_feat"
    )
    fig = px.scatter(df, x=feat, y=TARGET, opacity=0.55, title=f"{feat} vs {TARGET}")
    st.plotly_chart(fig)


# ───────────────────────── 페이지 2: 수용해도 예측 ─────────────────────────
def page_predict(df: pd.DataFrame):
    st.header("🔬 수용해도 예측")
    st.caption("분자 구조 기술자 4개를 입력하면 선형 회귀 모델이 logS를 예측합니다.")
    st.info("수용해도의 의미와 각 분자 구조 값의 영향이 궁금하면 사이드바의 **📖 수용해도 해설**을 확인하세요.")

    res = train_model(df)
    model, features = res["model"], res["features"]

    st.subheader("분자 구조 입력")
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(features):
        lo, hi = float(df[f].min()), float(df[f].max())
        mid = float(df[f].median())
        with cols[i % 2]:
            values[f] = st.slider(
                f"{f}  —  {COLUMN_INFO[f][1]}",
                min_value=lo,
                max_value=hi,
                value=mid,
                step=(hi - lo) / 200 or 0.01,
            )

    x_new = pd.DataFrame([values])[features]
    log_s = float(model.predict(x_new)[0])
    mol_l = 10**log_s
    g_l = mol_l * values["MolWt"]

    m1, m2, m3 = st.columns(3)
    m1.metric("예측 logS", f"{log_s:.2f}")
    m2.metric("수용해도 (mol/L)", f"{mol_l:.3g}")
    m3.metric("수용해도 (g/L)", f"{g_l:.3g}")
    st.caption("g/L 값은 mol/L × 분자량(MolWt)으로 환산한 값입니다.")

    st.divider()
    st.subheader("모델 성능")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("R² (학습)", f"{res['r2_train']:.3f}")
    p2.metric("R² (테스트)", f"{res['r2_test']:.3f}")
    p3.metric("MSE (학습)", f"{res['mse_train']:.3f}")
    p4.metric("MSE (테스트)", f"{res['mse_test']:.3f}")
    st.caption("학습:테스트 = 8:2 분할, 선형 회귀(LinearRegression), random_state=42")

    left, right = st.columns(2)

    with left:
        plot_df = pd.concat(
            [
                pd.DataFrame(
                    {"실측 logS": res["y_train"], "예측 logS": res["pred_train"], "구분": "학습"}
                ),
                pd.DataFrame(
                    {"실측 logS": res["y_test"], "예측 logS": res["pred_test"], "구분": "테스트"}
                ),
            ]
        )
        fig = px.scatter(
            plot_df, x="실측 logS", y="예측 logS", color="구분", opacity=0.6,
            title="실측 vs 예측",
        )
        lo = float(min(plot_df["실측 logS"].min(), plot_df["예측 logS"].min()))
        hi = float(max(plot_df["실측 logS"].max(), plot_df["예측 logS"].max()))
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                       line=dict(dash="dash", color="gray"))
        )
        fig.add_trace(
            go.Scatter(x=[log_s], y=[log_s], mode="markers", name="입력 분자(예측)",
                       marker=dict(size=14, symbol="star", color="black"))
        )
        st.plotly_chart(fig)

    with right:
        coef = pd.DataFrame({"특성": features, "계수": model.coef_})
        fig = px.bar(coef, x="특성", y="계수", title="회귀 계수 (부호: 양수면 logS 증가)")
        st.plotly_chart(fig)
        st.caption(f"절편: {model.intercept_:.3f}")


# ───────────────────────── 페이지 3: 수용해도 해설 ─────────────────────────
FEATURE_GUIDE = {
    "MolLogP": {
        "title": "MolLogP — 소수성(물을 싫어하는 정도)",
        "meaning": (
            "분자가 기름(옥탄올)과 물 중 어느 쪽에 더 잘 섞이는지를 나타내는 로그 값입니다. "
            "값이 클수록 기름을 좋아하고 물을 싫어하는(소수성) 분자이고, 작거나 음수이면 "
            "물과 잘 어울리는(친수성) 분자입니다."
        ),
        "effect": (
            "일반적으로 가장 강한 영향을 줍니다. MolLogP가 클수록 물 분자와 수소결합을 맺기 어려워 "
            "수용해도(logS)가 **낮아지는** 경향이 있습니다."
        ),
    },
    "MolWt": {
        "title": "MolWt — 분자량",
        "meaning": "분자를 이루는 모든 원자의 질량 합(g/mol)으로, 분자의 크기를 대신 보여 줍니다.",
        "effect": (
            "분자가 클수록 물 분자가 둘러싸서 녹이기 어렵고, 고체 상태에서 서로 단단히 뭉치는 경향도 "
            "있어 수용해도가 **낮아지는** 경향이 있습니다. 다만 MolLogP와 함께 커지는 경우가 많아 "
            "두 효과가 섞여 있습니다."
        ),
    },
    "NumRotatableBonds": {
        "title": "NumRotatableBonds — 회전 가능한 결합 수",
        "meaning": (
            "자유롭게 돌 수 있는 단일 결합의 개수로, 분자의 유연성을 나타냅니다. "
            "0이면 단단한(rigid) 분자, 클수록 구부러지기 쉬운 사슬형 분자입니다."
        ),
        "effect": (
            "영향이 상대적으로 **작고 방향이 일정하지 않습니다.** 유연한 분자는 결정으로 쌓이기 "
            "어려워 녹기 쉬울 수도 있지만, 사슬이 길어지면 소수성이 커져 반대로 작용할 수도 있습니다."
        ),
    },
    "AromaticProportion": {
        "title": "AromaticProportion — 방향족 원자 비율",
        "meaning": (
            "분자의 무거운 원자(수소 제외) 중 벤젠 고리 같은 방향족 구조에 속한 원자의 비율입니다. "
            "0이면 방향족 구조가 없고, 1에 가까울수록 대부분이 방향족입니다."
        ),
        "effect": (
            "방향족 고리는 평평하고 소수성이며 결정 상태에서 층층이 잘 쌓입니다. 그래서 비율이 높을수록 "
            "수용해도가 **낮아지는** 경향이 있습니다."
        ),
    },
}


def page_explain(df: pd.DataFrame):
    st.header("📖 수용해도 해설")

    # 1) 수용해도란?
    st.subheader("1. 수용해도란?")
    st.markdown(
        """
**수용해도(aqueous solubility)** 는 일정한 온도에서 물 일정량에 **최대로 녹을 수 있는 물질의 양**입니다.
더 녹일 수 없는 상태(포화 용액)의 농도로 나타내며, 이 앱에서는 **mol/L** 단위를 씁니다.

수용해도는 값의 범위가 매우 넓어서 보통 로그를 취해 **logS = log₁₀(mol/L)** 로 나타냅니다.
logS가 1 줄어들 때마다 녹는 양은 **10배씩 줄어듭니다.**
신약 후보 물질이 몸에 흡수되는 정도나 오염 물질이 물에 퍼지는 정도를 가늠할 때 중요한 값입니다.
        """
    )
    scale = pd.DataFrame(
        {
            "logS": [0, -2, -4, -6, -8],
            "mol/L": ["1", "0.01", "0.0001", "0.000001", "0.00000001"],
            "대략적인 해석": ["매우 잘 녹음", "잘 녹는 편", "조금 녹음", "잘 안 녹음", "거의 안 녹음"],
        }
    )
    st.dataframe(scale, hide_index=True)
    st.caption("해석 구간은 이해를 돕기 위한 대략적인 기준이며, 분야마다 기준이 조금씩 다릅니다.")

    fig = px.histogram(df, x=TARGET, nbins=40, title="이 데이터에서 logS 분포")
    fig.update_layout(yaxis_title="화합물 수", bargap=0.05)
    st.plotly_chart(fig)

    # 2) 데이터로 확인한 영향력
    st.subheader("2. 분자 구조 값이 수용해도에 미치는 영향")
    features = [c for c in df.columns if c != TARGET]

    corr = df[features].corrwith(df[TARGET])
    z = (df[features] - df[features].mean()) / df[features].std()
    model = LinearRegression().fit(z, df[TARGET])
    std_coef = pd.Series(model.coef_, index=features)

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            x=features, y=corr.values, title="logS와의 상관계수",
            labels={"x": "분자 구조 값", "y": "상관계수"},
        )
        fig.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig)
    with right:
        fig = px.bar(
            x=features, y=std_coef.values, title="표준화 회귀 계수 (다른 값을 고정했을 때의 영향)",
            labels={"x": "분자 구조 값", "y": "logS 변화량 / 표준편차 1"},
        )
        st.plotly_chart(fig)

    st.markdown(
        """
- **상관계수**는 한 값만 따로 봤을 때 logS와 얼마나 같이 움직이는지를 보여 줍니다.
- **표준화 회귀 계수**는 네 값을 같은 눈금(표준편차)으로 맞춘 뒤, 나머지를 고정하고 한 값만 표준편차 1만큼
  바꿨을 때 logS가 얼마나 변하는지를 보여 줍니다. 값의 크기를 비교해 **어느 구조 값이 더 큰 영향을 주는지** 가늠할 수 있습니다.
- 두 그래프에서 부호가 다르거나 크기가 크게 다르면, 구조 값들끼리 서로 연관되어 있다는 뜻입니다
  (예: 분자량이 큰 분자는 MolLogP도 큰 경향).
- 이 결과는 데이터에서 나타난 **경향**이며, 원인을 증명하는 것은 아닙니다.
        """
    )

    # 3) 각 구조 값의 의미
    st.subheader("3. 각 분자 구조 값의 의미")
    for f in features:
        info = FEATURE_GUIDE.get(f)
        if info is None:
            continue
        with st.expander(info["title"], expanded=(f == features[0])):
            st.markdown(f"**의미**  \n{info['meaning']}")
            st.markdown(f"**수용해도에 미치는 영향**  \n{info['effect']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("logS와의 상관계수", f"{corr[f]:+.2f}")
            c2.metric("표준화 회귀 계수", f"{std_coef[f]:+.2f}")
            c3.metric("데이터 범위", f"{df[f].min():.2f} ~ {df[f].max():.2f}")
            fig = px.scatter(df, x=f, y=TARGET, opacity=0.5, title=f"{f} vs {TARGET}")
            st.plotly_chart(fig)

    st.info(
        "정리: 물을 싫어하는 성질(MolLogP↑), 큰 분자(MolWt↑), 평평한 방향족 구조(AromaticProportion↑)가 "
        "클수록 대체로 물에 잘 녹지 않습니다. 위 숫자는 이 데이터에서 계산한 값입니다."
    )


# ───────────────────────── 페이지 4: 직접 해보기 ─────────────────────────
WATER_ML = 100.0  # 비커 속 물의 양
WATER_L = WATER_ML / 1000
WATER_RGB = (205, 232, 250)  # 맑은 물
# 물질마다 고유한 색 (이모지, 색 이름, 가장 진할 때의 RGB)
PALETTE = [
    ("🔴", "빨강", (214, 40, 40)),
    ("🟠", "주황", (240, 130, 20)),
    ("🟡", "노랑", (235, 200, 20)),
    ("🟢", "초록", (40, 160, 70)),
    ("🔵", "파랑", (30, 100, 220)),
    ("🟣", "보라", (140, 70, 190)),
    ("🟤", "갈색", (125, 80, 45)),
    ("⚫", "검정", (40, 40, 40)),
]


@st.cache_data
def lab_substances(df: pd.DataFrame) -> pd.DataFrame:
    """logS가 고르게 퍼지도록 실제 화합물 8개를 뽑아 '물질 A~H'로 이름 붙이고 고유 색을 준다."""
    ordered = df.sort_values(TARGET).reset_index(drop=True)
    count = len(PALETTE)
    positions = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    picked = ordered.iloc[positions].reset_index(drop=True)
    order = list(range(count))
    random.Random(7).shuffle(order)  # 목록 순서만으로 용해도를 짐작하지 못하게 섞는다
    picked = picked.iloc[order].reset_index(drop=True)
    picked["물질명"] = [f"물질 {chr(65 + i)}" for i in range(count)]
    picked["색번호"] = list(range(count))
    return picked


def substance_color(row: pd.Series) -> tuple:
    """(이모지, 색 이름, RGB)"""
    return PALETTE[int(row["색번호"])]


def lab_result(row: pd.Series, total_mass: float) -> dict:
    """비커(물 100 mL)에 total_mass g을 넣었을 때의 용해 결과."""
    s_mol_l = 10 ** float(row[TARGET])
    s_g_l = s_mol_l * float(row["MolWt"])
    limit_g = s_g_l * WATER_L  # 100 mL에 녹을 수 있는 최대 질량
    dissolved = min(total_mass, limit_g)
    undissolved = total_mass - dissolved
    return {
        "s_mol_l": s_mol_l,
        "s_g_l": s_g_l,
        "limit_g": limit_g,
        "dissolved": dissolved,
        "undissolved": undissolved,
        "conc_g_l": dissolved / WATER_L,
    }


def fmt_g(x: float) -> str:
    if x == 0:
        return "0"
    if x < 0.001 or x >= 10000:
        return f"{x:.2e}"
    return f"{x:.3g}"


WATER_TOP = 8.0  # 수면 높이 (그림 좌표)
SPAWN_Y = 12.3  # 용질이 떨어지기 시작하는 높이
ANIM_FRAMES = 100  # 총 프레임 수
FRAME_MS = 50  # 프레임 간격(ms): 100프레임 ≈ 5초 안에 낙하·수면파·용해·색 변화가 모두 끝난다
ANIM_PARTICLES = 18  # 한 번 넣을 때 떨어뜨리는 알갱이 수


def lerp_rgb(t: float, deep: tuple) -> tuple:
    return tuple(round(a + (b - a) * t) for a, b in zip(WATER_RGB, deep))


def water_t(conc_g_l: float) -> float:
    """녹은 농도(g/L) → 물 색 진하기(0~1). 1e-4 g/L 이하는 투명, 10 g/L 이상은 가장 진함 (로그 눈금)."""
    return min(max((math.log10(max(conc_g_l, 1e-12)) + 4) / 5, 0.0), 1.0)


def pile_count(frac_undissolved: float) -> int:
    return 0 if frac_undissolved < 0.005 else math.ceil(frac_undissolved * 70)


def pile_band(frac_undissolved: float) -> float:
    return 0.25 + 1.6 * frac_undissolved


def pile_positions(seed: int, n: int, band: float) -> tuple:
    """바닥에 쌓인 알갱이 위치. 같은 seed면 앞쪽 알갱이의 위치가 항상 같다."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        xs.append(rng.uniform(2.45, 7.55))
        ys.append(1.25 + rng.random() * band)
    return xs, ys


def _rgba(rgb: tuple, alpha: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _darker(rgb: tuple, d: int = 40) -> tuple:
    return tuple(max(c - d, 0) for c in rgb)


def is_dark_theme() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _glass(fig, title, subtitle):
    """비커 유리, 눈금, 실험대, 제목 등 움직이지 않는 부분."""
    for ml in (25, 50, 75, 100):  # 눈금 (100 mL = 물 높이)
        y = 1 + (WATER_TOP - 1) * ml / WATER_ML
        fig.add_shape(type="line", x0=7.35, x1=8, y0=y, y1=y, line=dict(color="#7b8794", width=2))
        fig.add_annotation(x=8.15, y=y, text=f"{ml} mL", showarrow=False, xanchor="left",
                           font=dict(size=11))
    fig.add_shape(
        type="path",
        path="M 1.6,11.9 L 2,11.5 L 2,2 Q 2,1 3,1 L 7,1 Q 8,1 8,2 L 8,11.5 L 8.4,11.9",
        line=dict(color="#6b7785", width=4),
    )
    fig.add_shape(type="line", x0=0.5, x1=9.5, y0=0.9, y1=0.9, line=dict(color="#9aa5b1", width=3))
    fig.add_annotation(x=5, y=12.6, text=f"<b>{title}</b>", showarrow=False, font=dict(size=18))
    fig.add_annotation(x=5, y=0.2, text=subtitle, showarrow=False, font=dict(size=13))

    fig.update_xaxes(range=[0, 10], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0, 13.2], visible=False, fixedrange=True, scaleanchor="x")
    fig.update_layout(
        height=540, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_beaker(t, frac_undissolved, seed, title, subtitle, deep=(30, 100, 220), pile_seed=0):
    """움직이지 않는 비커 그림. t: 물 색 진하기(0~1), deep: 물질 고유색."""
    rng = random.Random(seed)
    r, g, b = lerp_rgb(t, deep)
    dr, dg, db = deep

    fig = go.Figure()

    # 물
    fig.add_shape(
        type="path",
        path=(
            f"M 2.08,{WATER_TOP} L 2.08,2 Q 2.08,1.08 3,1.08 L 7,1.08 "
            f"Q 7.92,1.08 7.92,2 L 7.92,{WATER_TOP} Z"
        ),
        fillcolor=f"rgba({r},{g},{b},0.82)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="line", x0=2.08, x1=7.92, y0=WATER_TOP, y1=WATER_TOP,
        line=dict(color=_rgba(_darker((r, g, b)), 1), width=2),
    )

    # 녹은 입자 (진할수록 많이)
    for _ in range(int(t * 70)):
        fig.add_trace(go.Scatter(
            x=[rng.uniform(2.3, 7.7)], y=[rng.uniform(1.5, WATER_TOP - 0.2)],
            mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=4, color=f"rgba({dr},{dg},{db},0.6)"),
        ))

    # 녹지 않고 가라앉은 알갱이
    n_solid = pile_count(frac_undissolved)
    if n_solid:
        xs, ys = pile_positions(pile_seed, n_solid, pile_band(frac_undissolved))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(symbol="square", size=9, color=f"rgba({dr},{dg},{db},0.95)",
                        line=dict(color="white", width=1)),
        ))

    return _glass(fig, title, subtitle)


def make_drop_animation(deep, n_dissolve, n_keep, t_prev, t_new, pile_prev, band_new,
                        seed, title, subtitle, dark=False):
    """용질이 떨어지는 장면을 plotly 프레임 애니메이션으로 만든다.

    - 공중 낙하: 중력으로 점점 빨라진다.
    - 수면 충돌: 닿는 순간 수면파가 바깥으로 퍼지며 잦아든다.
    - 물속 감속: 물의 저항으로 속도가 줄어 일정한 느린 속도로 가라앉는다.
    - 녹을 용질: 수면 근처에서 크기가 서서히 줄어 사라진다.
    - 녹지 않을 용질: 크기 그대로 바닥까지 가라앉는다.
    - 물 색: 첫 충돌부터 마지막 용질이 사라질 때까지 이전 색에서 새 색으로 서서히 변한다.
    """
    rng = random.Random(seed)
    F = ANIM_FRAMES
    grav = 0.03  # 공기 중 가속도 (좌표/프레임²)
    big = 12  # 녹을 용질의 처음 크기(px)

    # 1) 알갱이마다 운동을 프레임별로 시뮬레이션
    parts = []
    for i in range(n_dissolve + n_keep):
        dissolve = i < n_dissolve
        x0 = rng.uniform(3.0, 7.0)
        delay = rng.randint(0, 14)  # 떨어지기 시작하는 시점(프레임)
        phase = rng.uniform(0, 6.28)
        rest = 1.25 + rng.random() * band_new  # 바닥에 멈출 높이
        shrink = rng.randint(28, 45)  # 녹는 데 걸리는 프레임 수
        drag, v_term = (0.40, 0.006) if dissolve else (0.80, 0.12)  # 녹을 용질은 수면 근처에 머문다
        x, y, v = x0, SPAWN_Y, 0.0
        landed, hit = False, None
        xs, ys = [], []
        for f in range(F):
            if f >= delay and not landed:
                if y > WATER_TOP:  # 공기 중: 중력 가속
                    v += grav
                    y -= v
                    if y <= WATER_TOP:
                        hit = f
                else:  # 물속: 저항으로 감속 → 종단 속도
                    v = v * drag + (1 - drag) * v_term
                    y -= v
                    x = x0 + 0.12 * math.sin(0.3 * (f - hit) + phase)
                    if not dissolve and y <= rest:
                        y, landed = rest, True
            xs.append(x)
            ys.append(y)
        parts.append({"dissolve": dissolve, "xs": xs, "ys": ys, "hit": hit, "shrink": shrink})

    def size_at(p, f):
        if p["hit"] is None or f < p["hit"]:
            return big
        return big * max(0.0, 1 - (f - p["hit"]) / p["shrink"])

    # 2) 수면파: 충돌 지점에서 바깥으로 퍼지며 잦아드는 파동
    impacts = [(p["hit"], p["xs"][p["hit"]]) for p in parts if p["hit"] is not None]
    sx = [2.08 + 5.84 * k / 60 for k in range(61)]

    def wave(f, x):
        h = 0.0
        for t_i, x_i in impacts:
            dt = f - t_i
            if dt < 0:
                continue
            dist = abs(x - x_i)
            front = 0.14 * dt  # 파면이 퍼지는 속도
            if dist > front:
                continue
            edge = min(1.0, (front - dist) / 0.5)
            h += 0.10 * edge * math.exp(-dt / 20) * math.exp(-0.2 * dist) * math.sin(3.0 * dist - 0.42 * dt)
        return max(-0.35, min(0.35, h))

    # 3) 물 색 변화 구간: 첫 충돌 ~ 마지막 녹을 용질이 사라질 때
    first_hit = min((p["hit"] for p in parts if p["hit"] is not None), default=20)
    last_gone = max(
        (p["hit"] + p["shrink"] for p in parts if p["dissolve"] and p["hit"] is not None),
        default=first_hit + 40,
    )
    c0, c1 = first_hit, max(last_gone, first_hit + 20)

    def smooth(u):
        u = min(max(u, 0.0), 1.0)
        return u * u * (3 - 2 * u)

    # 물속에 퍼진 작은 입자(색이 짙어질수록 잘 보임)
    drng = random.Random(seed + 1)
    n_dots = int(t_new * 70)
    dots_x = [drng.uniform(2.3, 7.7) for _ in range(n_dots)]
    dots_y = [drng.uniform(1.5, WATER_TOP - 0.3) for _ in range(n_dots)]

    # 4) 프레임별로 바뀌는 속성 (트레이스 번호 0, 1, 2, 4, 5)
    dyn_idx = [0, 1, 2, 4, 5]
    bottom_x = [7.92, 7.7, 7.0, 3.0, 2.3, 2.08]
    bottom_y = [2.0, 1.25, 1.08, 1.08, 1.25, 2.0]
    keep_parts = [p for p in parts if not p["dissolve"]]
    dis_parts = [p for p in parts if p["dissolve"]]

    def attrs(f):
        t_f = t_prev + (t_new - t_prev) * smooth((f - c0) / (c1 - c0))
        rgb = lerp_rgb(t_f, deep)
        surf = [WATER_TOP + wave(f, x) for x in sx]
        return [
            dict(x=sx + bottom_x, y=surf + bottom_y, fillcolor=_rgba(rgb, 0.82)),
            dict(x=sx, y=surf, line=dict(color=_rgba(_darker(rgb), 1), width=2)),
            dict(marker=dict(opacity=0.6 * (t_f / t_new) if t_new > 0 else 0.0)),
            dict(x=[p["xs"][f] for p in keep_parts], y=[p["ys"][f] for p in keep_parts]),
            dict(x=[p["xs"][f] for p in dis_parts], y=[p["ys"][f] for p in dis_parts],
                 marker=dict(size=[size_at(p, f) for p in dis_parts])),
        ]

    styles = {
        0: dict(mode="lines", fill="toself", line=dict(width=0), hoverinfo="skip", showlegend=False),
        1: dict(mode="lines", hoverinfo="skip", showlegend=False),
        2: dict(mode="markers", x=dots_x, y=dots_y, hoverinfo="skip", showlegend=False,
                marker=dict(size=4, color=_rgba(deep, 1))),
        4: dict(mode="markers", name="녹지 않을 용질", hoverinfo="skip", showlegend=n_keep > 0,
                marker=dict(symbol="square", size=9, color=_rgba(deep, 1),
                            line=dict(color="white", width=1))),
        5: dict(mode="markers", name="녹을 용질", hoverinfo="skip", showlegend=n_dissolve > 0,
                marker=dict(symbol="circle", color=_rgba(deep, 0.75),
                            line=dict(color=_rgba(deep, 1), width=1.5))),
    }

    first = attrs(0)
    data = [None] * 6
    for k, i in enumerate(dyn_idx):
        trace = go.Scatter(**styles[i])
        trace.update(first[k])
        data[i] = trace
    pile_x, pile_y = pile_prev  # 이미 바닥에 쌓여 있던 알갱이 (움직이지 않음)
    data[3] = go.Scatter(
        x=pile_x, y=pile_y, mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(symbol="square", size=9, color=_rgba(deep, 0.95), line=dict(color="white", width=1)),
    )

    fig = go.Figure(data=data)
    fig.frames = [
        go.Frame(data=[go.Scatter(**a) for a in attrs(f)], traces=dyn_idx, name=str(f))
        for f in range(F)
    ]

    _glass(fig, title, subtitle)
    fig.update_layout(
        showlegend=True,
        font=dict(color="#fafafa" if dark else "#31333f"),
        legend=dict(x=0.0, y=1.0, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=1.0, y=1.0, xanchor="right", yanchor="top",
            buttons=[dict(
                label="▶ 다시 재생", method="animate",
                args=[None, dict(frame=dict(duration=FRAME_MS, redraw=True),
                                 transition=dict(duration=0), fromcurrent=False, mode="immediate")],
            )],
        )],
    )
    return fig


def page_lab(df: pd.DataFrame):
    st.header("🧫 직접 해보기")
    st.caption(
        f"물질과 질량을 고른 뒤 버튼을 눌러 물 {WATER_ML:.0f} mL가 든 비커에 넣어 보세요. "
        "물질의 실제 수용해도(logS)에 따라 물의 색과 바닥에 가라앉는 알갱이가 달라집니다."
    )

    subs = lab_substances(df)
    ss = st.session_state
    ss.setdefault("lab_sid", None)  # 비커에 들어 있는 물질 (subs 행 번호)
    ss.setdefault("lab_mass", 0.0)  # 이 비커에 넣은 누적 질량(g)
    ss.setdefault("lab_beaker", 1)  # 비커 번호
    ss.setdefault("lab_log", [])  # 실험 기록
    ss.setdefault("lab_msg", None)  # 마지막 결과 안내 (종류, 문장)

    # 앱을 수정한 뒤 이전 세션에 남은 값이 물질 목록 범위를 벗어나면 초기화한다
    if ss.lab_sid is not None and ss.lab_sid not in subs.index:
        ss.lab_sid, ss.lab_mass, ss.lab_msg = None, 0.0, None
    if "lab_choice" in ss and ss.lab_choice not in subs.index:
        del ss["lab_choice"]

    left, mid, right = st.columns([1.15, 2, 1.25], gap="large")

    # ── 왼쪽: 물질 선택 + 질량 + 버튼 ──
    with left:
        st.subheader("1. 물질 선택")
        sid = st.radio(
            "물질",
            list(subs.index),
            format_func=lambda i: (
                f"{substance_color(subs.loc[i])[0]} {subs.at[i, '물질명']}  ·  분자량 {subs.at[i, 'MolWt']:.0f}"
                f"  ·  MolLogP {subs.at[i, 'MolLogP']:.1f}"
            ),
            key="lab_choice",
            label_visibility="collapsed",
        )
        st.subheader("2. 질량 선택")
        mass = st.slider("이번에 넣을 질량 (g)", 0.1, 50.0, 5.0, 0.1, key="lab_slider")
        add = st.button("🧪 비커에 넣기", type="primary")
        reset = st.button("🔄 새 비커로 교체")
        st.caption("같은 물질을 또 넣으면 같은 비커에 질량이 쌓이고, 다른 물질을 넣으면 새 비커로 바뀝니다.")

        if reset:
            ss.lab_beaker += 1
            ss.lab_sid, ss.lab_mass = None, 0.0
            ss.lab_msg = ("info", "깨끗한 새 비커를 꺼냈어요.")
            if "lab_anim" in ss:
                del ss["lab_anim"]
            st.toast("새 비커로 교체했어요", icon="🔄")

        if add:
            if ss.lab_sid is not None and ss.lab_sid != sid:
                ss.lab_beaker += 1
                ss.lab_mass = 0.0
                st.toast(f"다른 물질이라 비커 #{ss.lab_beaker}로 바꿨어요", icon="🔄")
            ss.lab_sid = sid
            row_new = subs.loc[sid]
            prev_total = ss.lab_mass  # 이번에 넣기 전까지 이 비커에 들어 있던 양
            prev_res = lab_result(row_new, prev_total) if prev_total > 0 else None
            ss.lab_mass += mass
            res = lab_result(row_new, ss.lab_mass)
            ratio = res["dissolved"] / ss.lab_mass

            # 이번에 넣은 양 중 녹을 몫과 녹지 않을 몫을 나눠 애니메이션용 알갱이 수를 정한다
            added_dissolved = res["dissolved"] - (prev_res["dissolved"] if prev_res else 0.0)
            share = min(max(added_dissolved / mass, 0.0), 1.0)
            n_dis = round(ANIM_PARTICLES * share)
            if share > 0.02 and n_dis == 0:
                n_dis = 1
            ss.lab_anim = {
                "sid": sid,
                "beaker": ss.lab_beaker,
                "n_dis": n_dis,
                "n_keep": ANIM_PARTICLES - n_dis,
                "t_prev": water_t(prev_res["conc_g_l"]) if prev_res else 0.0,
                "t_new": water_t(res["conc_g_l"]),
                "frac_prev": prev_res["undissolved"] / prev_total if prev_res else 0.0,
                "frac_new": res["undissolved"] / ss.lab_mass,
                "seed": ss.lab_beaker * 1000 + len(ss.lab_log),
            }
            if res["undissolved"] <= 0:
                kind, text, icon = "success", "넣은 물질이 모두 녹았어요 (불포화).", "✅"
            elif ratio < 0.01:
                kind, text, icon = "warning", "거의 녹지 않고 바닥에 가라앉았어요.", "🪨"
            else:
                kind, text, icon = (
                    "warning",
                    f"포화! {fmt_g(res['dissolved'])} g만 녹고 {fmt_g(res['undissolved'])} g은 가라앉았어요.",
                    "⚠️",
                )
            ss.lab_msg = (kind, text)
            st.toast(text, icon=icon)
            ss.lab_log.append(
                {
                    "비커": f"#{ss.lab_beaker}",
                    "물질": subs.at[sid, "물질명"],
                    "넣은 질량(g)": mass,
                    "누적 질량(g)": round(ss.lab_mass, 2),
                    "녹은 질량(g)": float(f"{res['dissolved']:.3g}"),
                    "가라앉은 질량(g)": float(f"{res['undissolved']:.3g}"),
                }
            )

    # ── 현재 비커 상태 계산 ──
    active = ss.lab_sid
    deep = PALETTE[0][2]
    if active is None:
        t, frac, title, subtitle, res = 0.0, 0.0, f"비커 #{ss.lab_beaker}", f"물 {WATER_ML:.0f} mL (비어 있음)", None
    else:
        row = subs.loc[active]
        deep = substance_color(row)[2]
        res = lab_result(row, ss.lab_mass)
        t = water_t(res["conc_g_l"])
        frac = res["undissolved"] / ss.lab_mass if ss.lab_mass > 0 else 0.0
        title = f"비커 #{ss.lab_beaker}"
        subtitle = f"{substance_color(row)[0]} {row['물질명']} {ss.lab_mass:.1f} g + 물 {WATER_ML:.0f} mL"

    # ── 가운데: 비커 ──
    with mid:
        anim = ss.get("lab_anim")
        if "lab_anim" in ss:
            del ss["lab_anim"]  # 버튼을 누른 직후 한 번만 재생한다

        if anim is not None and active is not None and anim["sid"] == active:
            pile_prev = pile_positions(anim["beaker"], pile_count(anim["frac_prev"]), pile_band(anim["frac_prev"]))
            fig = make_drop_animation(
                deep, anim["n_dis"], anim["n_keep"], anim["t_prev"], anim["t_new"],
                pile_prev, pile_band(anim["frac_new"]), anim["seed"], title, subtitle,
                dark=is_dark_theme(),
            )
            try:
                # 페이지가 열리자마자 프레임이 자동 재생되도록 HTML로 내보낸다
                html = fig.to_html(
                    include_plotlyjs="cdn", full_html=False, auto_play=True,
                    animation_opts=dict(
                        frame=dict(duration=FRAME_MS, redraw=True),
                        transition=dict(duration=0), mode="immediate",
                    ),
                    default_width="100%", default_height="540px",
                    config={"displayModeBar": False},
                )
                components.html(html, height=550)
            except Exception:
                # 자동 재생을 쓸 수 없으면 일반 차트로 보여 주고 ▶ 버튼으로 재생한다
                st.plotly_chart(fig, config={"displayModeBar": False})
        else:
            fig = make_beaker(
                t, frac, seed=ss.lab_beaker * 1000 + len(ss.lab_log), title=title,
                subtitle=subtitle, deep=deep, pile_seed=ss.lab_beaker,
            )
            st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption(
            "물질을 넣으면 공중에서 떨어져 수면파를 만들고, 물속에서는 저항으로 느려집니다. "
            "녹을 용질(●)은 수면 근처에서 점점 작아지며 물이 그 물질의 고유색으로 서서히 변하고, "
            "녹지 않을 용질(■)은 바닥까지 가라앉습니다. 물이 진해지는 정도는 녹은 농도에 따라 달라집니다(로그 눈금)."
        )

    # ── 오른쪽: 결과 ──
    with right:
        st.subheader("3. 결과")
        if active is None:
            st.info("물질과 질량을 고르고 **비커에 넣기**를 눌러 보세요.")
        else:
            row = subs.loc[active]
            emoji, color_name, _ = substance_color(row)
            st.markdown(f"{emoji} **{row['물질명']}** ({color_name}색 물질) · 비커 #{ss.lab_beaker}")
            m1, m2 = st.columns(2)
            m1.metric("실측 logS", f"{row[TARGET]:.2f}")
            m2.metric("수용해도", f"{fmt_g(res['s_g_l'])} g/L")
            m3, m4 = st.columns(2)
            m3.metric(f"{WATER_ML:.0f} mL에 녹는 한계", f"{fmt_g(res['limit_g'])} g")
            m4.metric("누적 넣은 양", f"{ss.lab_mass:.1f} g")
            m5, m6 = st.columns(2)
            m5.metric("녹은 양", f"{fmt_g(res['dissolved'])} g")
            m6.metric("가라앉은 양", f"{fmt_g(res['undissolved'])} g")
            st.progress(
                min(max(res["dissolved"] / ss.lab_mass, 0.0), 1.0),
                text=f"넣은 양 중 녹은 비율 {res['dissolved'] / ss.lab_mass:.2%}",
            )
            if ss.lab_msg:
                getattr(st, ss.lab_msg[0])(ss.lab_msg[1])

            trained = train_model(df)
            pred = float(trained["model"].predict(pd.DataFrame([row[trained["features"]]]))[0])
            st.caption(f"참고: 선형 회귀 모델의 예측 logS는 {pred:.2f}입니다 (실측 {row[TARGET]:.2f}).")
            st.caption(
                "계산: 한계(g) = 10^logS × 분자량 × 0.1 L. "
                "넣은 양이 한계보다 크면 한계만큼만 녹고 나머지는 가라앉습니다."
            )

    # ── 아래: 실험 기록 ──
    if ss.lab_log:
        st.subheader("실험 기록")
        st.dataframe(pd.DataFrame(ss.lab_log[::-1]), hide_index=True)


# ───────────────────────── 메인 ─────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    data = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

PAGES = {
    "📊 데이터 정보": page_data_info,
    "🔬 수용해도 예측": page_predict,
    "📖 수용해도 해설": page_explain,
    "🧫 직접 해보기": page_lab,
}
page = st.sidebar.radio("메뉴", list(PAGES))
st.sidebar.caption(f"{APP_ICON} {APP_TITLE}")
PAGES[page](data)
