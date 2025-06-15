
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import os
import matplotlib.pyplot as plt

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
st.set_page_config(page_title="零售銷售預測平台", layout="wide")

# 自訂 CSS 用於凍結標題
st.markdown("""
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        padding: 1rem 0 0.5rem 0;
        z-index: 100;
        border-bottom: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # streamlit發佈網站
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    df = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    holiday_df = pd.read_csv("data/raw/holidays_events.csv", parse_dates=["date"])
    return df,holiday_df

df ,holiday_df = load_data()
df_all = df.copy()
store_options = sorted(df["store_nbr"].unique())

# 📌 預設：使用近三年資料
cutoff_date = df["date"].max() - pd.DateOffset(years=3)
df = df[df["date"] >= cutoff_date]

# 📅 預測與圖表區間
horizon_days_map = {"14 天": 14, "3 個月": 90, "6 個月": 180}
with st.sidebar:
    st.header("🔧 控制面板")
    lang = st.radio("🌐 語言 Language", ["中文", "English"], index=0)
    stores = st.multiselect("🏪 選擇店鋪", store_options, default=[1, 2])
    horizon_label = st.selectbox("🔮 預測未來區間", list(horizon_days_map.keys()), index=2)

    st.markdown("---")
    st.subheader("🤖 AI 助理模式")
    ai_mode = st.selectbox("🔌 模式選擇", ["離線模式", "AI 模式"])
    ai_enabled = False
    if ai_mode == "AI 模式":
        password = st.text_input("🔑 請輸入密碼啟用 AI 模式", type="password")
        if password == "joanna0408demo":
            ai_enabled = True
        else:
            st.warning("密碼錯誤，AI 模式未啟用")

    
    st.title("🤖 AI 助理")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if ai_enabled:
        prompt = st.chat_input("針對預測結果提問...")

        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            summary = "\n".join([f"店鋪 {d['store']}: 平均 {d['mean']:.1f}, 異常 {d['anomalies']}, MAE {d['mae']:.1f}" for d in st.session_state["forecast_info"]])

            with st.spinner("AI 分析中..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"你是一位專業零售分析顧問。使用者目前分析的店鋪資料如下：\n{summary}"},
                            *st.session_state.chat
                        ]
                    )
                    reply = response.choices[0].message["content"]
                    st.session_state.chat.append({"role": "assistant", "content": reply})
                except Exception as e:
                    reply = f"❗ 發生錯誤：{e}"
                    st.session_state.chat.append({"role": "assistant", "content": reply})

    # 🧠 自動 AI 建議提示（無輸入時）
    if ai_enabled and not prompt and "forecast_info" in st.session_state:
        auto_summary = "\n".join([f"店鋪 {d['store']}: 平均 {d['mean']:.1f}, MAE {d['mae']:.1f}, 異常天數 {d['anomalies']}" for d in st.session_state["forecast_info"]])
        auto_prompt = f"你是一位區域店長，請根據以下資料提出 3 個營運改善建議，供高層決策參考：\n{auto_summary}"
        with st.spinner("AI 自動分析中..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"你是一位零售分析顧問，請根據以下資料提出觀察重點與建議：\n{auto_summary}"},
                        {"role": "user", "content": auto_prompt}
                    ]
                )
                reply = response.choices[0].message["content"]
                st.markdown("### 🤖 自動建議卡片")
                st.info(reply)
            except Exception as e:
                st.warning(f"自動建議產生失敗：{e}")

    else:
        st.info("目前為離線模式，僅提供基本摘要，不會使用 OpenAI API。")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


horizon_days = horizon_days_map[horizon_label]
chart_start = df["date"].max() - pd.DateOffset(months=6)
chart_end = df["date"].max() + pd.Timedelta(days=horizon_days)

def t(zh, en):
    return zh if lang == "中文" else en

# 📌 凍結標題區
st.title(t("📊 零售銷售預測分析", "📊 Retail Sales Forecast Analysis"))
st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

# 主欄與 AI 助理欄分欄
col1, col2  = st.columns(2)
with col1:
    st.markdown(t(
        f"""📦 **功能特色：**
- 使用 Prophet 與 XGBoost 雙模型預測銷售
- 訓練資料區間：**{cutoff_date.date()} ~ {df['date'].max().date()}**
- 預測區間：**{horizon_label}**
""",
        f"""📦 **Features:**
- Dual models: Prophet & XGBoost
- Training data: {cutoff_date.date()} ~ {df['date'].max().date()}
- Forecast horizon: {horizon_label}
"""
    ))
with col2:
    st.markdown(t(
        """📘 **指標說明**：
- **平均銷售**：觀察期間每日平均銷售（不含 0）
- **標準差**：銷售波動程度
- **異常天數**：超過 ±3σ 的天數
- **趨勢評估**：無異常為穩定，有為波動
""",
        """📘 **Metric Description**:
- **Average Sales**: Daily mean (excluding 0)
- **Std Dev**: Sales variability
- **Anomaly Days**: Outliers beyond ±3σ
- **Trend**: 'Stable' or 'Fluctuating'
"""
    ))
st.markdown('</div>', unsafe_allow_html=True)



# with col3:
# 📈 多店鋪 Prophet 比較（提前）
if len(stores) > 1:
    st.subheader("📈 多店鋪預測比較（Prophet）")
    all_prophet_forecast = []
    for store in stores:
        df_store_tmp = df[df["store_nbr"] == store].sort_values("date")
        df_prophet_tmp = df_store_tmp[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
        model_tmp = Prophet(daily_seasonality=True)
        model_tmp.fit(df_prophet_tmp)
        future_tmp = model_tmp.make_future_dataframe(periods=horizon_days)
        forecast_tmp = model_tmp.predict(future_tmp)
        forecast_tmp = forecast_tmp[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "sales"})
        forecast_tmp["store_nbr"] = store
        all_prophet_forecast.append(forecast_tmp)

    merged_prophet = pd.concat(all_prophet_forecast)
    chart_df = merged_prophet[merged_prophet["date"].between(chart_start, chart_end)]
    fig_multi = px.line(chart_df, x="date", y="sales", color="store_nbr", title="各店鋪 Prophet 預測比較")
    st.plotly_chart(fig_multi, use_container_width=True)

summary_lines = []

# 模擬每家店鋪的目前庫存量（可改為實際讀取資料）
mock_inventory = {store: np.random.randint(300, 1000) for store in stores}

all_prophet_forecast = []
all_forecast_info = []

# with col2:
for store in stores:
    st.subheader(t(f"🏪 店鋪 {store}", f"🏪 Store {store}"))
    df_store = df[df["store_nbr"] == store].sort_values("date")
    df_store_nz = df_store[df_store["sales"] > 0]
    df_prophet = df_store[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})

    # Prophet
    model_p = Prophet(daily_seasonality=True)
    model_p.fit(df_prophet)
    future = model_p.make_future_dataframe(periods=horizon_days)
    forecast = model_p.predict(future)
    forecast_prophet = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "sales"})
    forecast_prophet["model"] = "Prophet"
    forecast_prophet["store_nbr"] = store
    all_prophet_forecast.append(forecast_prophet)

    # Prophet MAE
    df_actual = df_all[df_all["store_nbr"] == store]
    merged = pd.merge(forecast_prophet, df_actual, on="date", how="inner")
    prophet_mae = mean_absolute_error(merged["sales_y"], merged["sales_x"]) if not merged.empty else np.nan

    # XGBoost
    df_xgb = df_store.copy()
    df_xgb["dayofweek"] = df_xgb["date"].dt.dayofweek
    df_xgb["month"] = df_xgb["date"].dt.month
    df_xgb["day"] = df_xgb["date"].dt.day
    df_xgb["year"] = df_xgb["date"].dt.year
    df_xgb["store_nbr"] = store
    X = df_xgb[["dayofweek", "month", "day", "year", "store_nbr"]]
    y = df_xgb["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model_x = XGBRegressor(n_estimators=100, max_depth=3)
    model_x.fit(X_train, y_train)
    y_pred = model_x.predict(X_test)

    xgb_result = pd.DataFrame({
        "date": df_xgb.loc[X_test.index, "date"],
        "sales": y_pred,
        "model": "XGBoost"
    })

    # Combine
    combined = pd.concat([
        forecast_prophet[forecast_prophet["date"].between(chart_start, chart_end)],
        xgb_result[xgb_result["date"].between(chart_start, chart_end)]
    ])

    fig = px.line(combined, x="date", y="sales", color="model",
                title=t(f"店鋪 {store} - 銷售預測比較", f"Store {store} - Forecast Comparison"))
    st.plotly_chart(fig, use_container_width=True)

    xgb_mae = mean_absolute_error(y_test, y_pred)
    df_summary = pd.DataFrame({
        "模型": ["Prophet", "XGBoost"],
        "平均預測": [forecast_prophet["sales"].mean(), xgb_result["sales"].mean()],
        "MAE": [prophet_mae, xgb_mae]
    })
    st.dataframe(df_summary)

    # 📊 營運狀況摘要
    observed = df_store_nz[df_store_nz["date"].between(chart_start, chart_end)]
    mean_sales = observed["sales"].mean()
    std_sales = observed["sales"].std()
    anomalies = observed[np.abs(observed["sales"] - mean_sales) > 3 * std_sales]
    anomaly_days = len(anomalies)
    trend = t("穩定", "Stable") if anomaly_days == 0 else t("波動", "Fluctuating")
    summary_text = f"🏪 店鋪 {store}：平均銷售 {mean_sales:.1f}，標準差 {std_sales:.1f}，異常天數 {anomaly_days}，趨勢評估：{trend}，MAE 誤差：{xgb_mae}"

    # 📦 庫存提醒與補貨建議
    forecast_period = forecast_prophet.tail(horizon_days)
    total_forecast = forecast_period["sales"].sum()
    inventory = mock_inventory[store]
    safety_factor = 1.1  # 安全係數 10%
    reorder_qty = max(int(total_forecast * safety_factor - inventory), 0)

    if inventory < total_forecast:
        st.error(f"⚠️ 店鋪 {store} 的預測銷售總量為 {total_forecast:.0f}，目前庫存僅 {inventory}，可能會發生缺貨！")
        if reorder_qty > 0:
            st.info(f"🔄 建議補貨約 {reorder_qty} 單位（含 10% 安全係數）")
    else:
        st.success(f"✅ 店鋪 {store} 庫存量足夠（預測需求：{total_forecast:.0f}，庫存：{inventory}）")

    st.markdown("#### 📊 " + t("營運狀況摘要", "Operational Summary"))
    st.markdown(summary_text)

    # 異常偵測 📉 異常銷售日分析
    st.subheader("📉 異常銷售日分析")
    anomaly_report = []
    for store in stores:
        df_store = df[df["store_nbr"] == store].sort_values("date")
        mean_sales = df_store['sales'].mean()
        std_sales = df_store['sales'].std()
        upper = mean_sales + 3 * std_sales
        lower = mean_sales - 3 * std_sales
        anomalies = df_store[(df_store['sales'] > upper) | (df_store['sales'] < lower)]
        for _, row in anomalies.iterrows():
            is_holiday = row['date'] in holiday_df['date'].values
            reason = "節日影響" if is_holiday else "可能異常波動"
            anomaly_report.append({
                '店鋪': store,
                '日期': row['date'].date(),
                '銷售額': row['sales'],
                '異常原因': reason
            })
    # 顯示異常報告
    if anomaly_report:
        st.dataframe(pd.DataFrame(anomaly_report))
    else:
        st.info("無明顯異常銷售紀錄。")
        
    # 提供給 AI 助理存取
    all_forecast_info.append({
            "store": store,
            "mean": mean_sales,
            "std": std_sales,
            "anomalies": anomaly_days,
            "trend": trend,
            "mae": xgb_mae
        })
    st.session_state["forecast_info"] = all_forecast_info


# 📊 預測模型與實際銷售總表格比較
compare_records = []

for store in stores:
    df_store = df_all[df_all["store_nbr"] == store].sort_values("date")
    actual_period = df_store[df_store["date"].between(chart_start, chart_end)]
    prophet_period = all_prophet_forecast[stores.index(store)].copy()
    prophet_period = prophet_period[prophet_period["date"].between(chart_start, chart_end)]

    df_prophet_avg = prophet_period["sales"].mean()
    df_actual_avg = actual_period["sales"].mean()

    # XGBoost 預測資料（從 all_forecast_info 中找 MAE）
    match = [d["mae"] for d in all_forecast_info if d["store"] == store]
    xgb_mae = round(match[0], 2) if match else np.nan

    compare_records.append({
        "店鋪": store,
        "實際平均銷售": round(df_actual_avg, 1),
        "Prophet 預測平均": round(df_prophet_avg, 1),
        "XGBoost MAE": round(xgb_mae, 2)
    })

df_compare = pd.DataFrame(compare_records)
st.subheader("📊 預測模型 vs 實際平均銷售總表")
st.dataframe(df_compare)


reorder_report = []

# 📌 彙整所有補貨建議
for store in stores:
    forecast_period = all_prophet_forecast[stores.index(store)].tail(horizon_days)
    total_forecast = forecast_period["sales"].sum()
    inventory = mock_inventory[store]
    safety_factor = 1.1
    reorder_qty = max(int(total_forecast * safety_factor - inventory), 0)

    reorder_report.append({
        "店鋪": store,
        "預測需求量": int(total_forecast),
        "目前庫存": inventory,
        "建議補貨量": reorder_qty
    })

# 匯出按鈕
reorder_df = pd.DataFrame(reorder_report)
st.download_button(
    label="📤 下載補貨建議清單 (CSV)",
    data=reorder_df.to_csv(index=False).encode("utf-8-sig"),
    file_name="reorder_suggestions.csv",
    mime="text/csv"
)
  
