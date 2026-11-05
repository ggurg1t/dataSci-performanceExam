import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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


# ───────────────────────── 메인 ─────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    data = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

page = st.sidebar.radio("메뉴", ["📊 데이터 정보", "🔬 수용해도 예측", "📖 수용해도 해설"])
st.sidebar.caption(f"{APP_ICON} {APP_TITLE}")

if page.startswith("📊"):
    page_data_info(data)
elif page.startswith("🔬"):
    page_predict(data)
else:
    page_explain(data)
