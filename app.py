import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import timedelta
from prophet import Prophet


@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/train.csv', parse_dates=['date'])
    df_daily = df.groupby(['date', 'store_nbr'])['sales'].sum().reset_index()

    # 讀取節日資料
    holidays = pd.read_csv('data/raw/holidays_events.csv', parse_dates=['date'])
    holidays = holidays[holidays['transferred'] == False]  # 排除已轉移的假日
    holidays = holidays[['date', 'description', 'type']]    # 精簡欄位
    return df_daily, holidays

def detect_anomalies(df):
    mean = df['sales'].mean()
    std = df['sales'].std()
    df['anomaly'] = (df['sales'] > mean + 3*std) | (df['sales'] < mean - 3*std)
    return df

# def forecast_sales(df, days=14):
#     df_sorted = df.sort_values('date')
#     df_sorted['sales_ma7'] = df_sorted['sales'].rolling(window=7).mean()
#     last_date = df_sorted['date'].max()
#     last_ma = df_sorted['sales_ma7'].dropna().iloc[-1] if not df_sorted['sales_ma7'].dropna().empty else df_sorted['sales'].iloc[-1]

#     future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
#     future_sales = [last_ma] * days

#     df_forecast = pd.DataFrame({
#         'date': future_dates,
#         'sales': future_sales,
#         'store_nbr': df_sorted['store_nbr'].iloc[0],
#         'forecast': True,
#         'anomaly': False
#     })

#     df_sorted['forecast'] = False
#     df_sorted['anomaly'] = False

#     return pd.concat([df_sorted, df_forecast], ignore_index=True)

#Facebook 的 Prophet 時間序列預測模型
def forecast_sales(df, days=14):
    df_sorted = df.sort_values('date')
    store_nbr = df_sorted['store_nbr'].iloc[0]

    # 轉換成 Prophet 格式
    prophet_df = df_sorted[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

    # 建立並訓練模型
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)

    # 產生未來日期
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # 合併原始與預測結果
    forecast_result = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'sales'})
    forecast_result['store_nbr'] = store_nbr
    forecast_result['forecast'] = True

    # 原始資料標記
    df_sorted['forecast'] = False

    df_combined = pd.concat([df_sorted[['date', 'sales', 'store_nbr', 'forecast']],
                             forecast_result[forecast_result['forecast'] == True]],
                            ignore_index=True)

    df_combined['anomaly'] = False  # 預測不進行異常分析
    return df_combined

def extract_anomaly_details(df, store_nbr, holidays_df):
    mean = df['sales'].mean()
    std = df['sales'].std()
    upper = mean + 3 * std
    lower = mean - 3 * std

    anomaly_df = df[df['anomaly']].copy()
    anomaly_df['異常類型'] = anomaly_df['sales'].apply(
        lambda x: '⚠️ 高異常值' if x > upper else '⚠️ 低異常值'
    )
    anomaly_df['store_nbr'] = store_nbr

    # 合併節日說明
    anomaly_df = anomaly_df.merge(holidays_df, how='left', on='date')

    # 原因欄位說明
    def explain(row):
        if pd.notna(row['description']):
            return f"{row['異常類型']}（節日：{row['description']}）"
        else:
            return f"{row['異常類型']}（超出 ±3σ）"

    anomaly_df['說明'] = anomaly_df.apply(explain, axis=1)

    return anomaly_df[['date', 'store_nbr', 'sales', '異常類型', '說明']]

def main():
    st.title("零售銷售趨勢互動展示 + 預測與異常偵測")

    df, holidays = load_data()
    store_list = sorted(df['store_nbr'].unique())

    selected_stores = st.multiselect("請選擇最多 6 間店鋪", options=store_list, default=store_list[:2])
    if len(selected_stores) > 6:
        st.warning("⚠️ 最多只能選擇 6 間店鋪")
        st.stop()

    max_date = df['date'].max()
    min_date = max_date - pd.DateOffset(years=2)
    default_start = max_date - pd.DateOffset(months=6)

    start_date, end_date = st.date_input("選擇銷售時間區間", [default_start, max_date], min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("開始日期不能大於結束日期")
        st.stop()

    plot_dfs = []
    summary = []
    all_anomaly_tables = []

    for store in selected_stores:
        store_df = df[(df['store_nbr'] == store) &
                      (df['date'] >= pd.to_datetime(start_date)) &
                      (df['date'] <= pd.to_datetime(end_date))].copy()
        if store_df.empty:
            continue

        store_df = detect_anomalies(store_df)
        store_all = forecast_sales(store_df, days=14)
        plot_dfs.append(store_all)

        anomaly_detail = extract_anomaly_details(store_df, store, holidays)
        if not anomaly_detail.empty:
            all_anomaly_tables.append(anomaly_detail)

        mean = store_df['sales'].mean()
        std = store_df['sales'].std()
        anomalies = store_df[store_df['anomaly']].shape[0]
        stability = "穩定" if std / mean < 0.3 else "波動大"
        summary.append(f"- 🏪 店鋪 {store}：平均銷售 {mean:.1f}，標準差 {std:.1f}，異常天數 {anomalies}，趨勢評估：**{stability}**")

    if not plot_dfs:
        st.warning("查無符合條件的資料")
        st.stop()

    df_plot = pd.concat(plot_dfs)

    fig = px.line(df_plot, x='date', y='sales', color='store_nbr',
                  line_dash='forecast',
                  title='📈 銷售趨勢與預測',
                  labels={'date': '日期', 'sales': '銷售額', 'store_nbr': '店鋪', 'forecast': '預測'})

    anomalies_df = df_plot[(df_plot['anomaly']) & (~df_plot['forecast'])]
    fig.add_scatter(x=anomalies_df['date'], y=anomalies_df['sales'],
                    mode='markers', marker=dict(color='red', size=8, symbol='x'),
                    name='異常銷售點')

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 營運狀況分析摘要")
    for line in summary:
        st.markdown(line)

    if all_anomaly_tables:
        st.subheader("📌 詳細異常銷售清單")
        full_anomaly_df = pd.concat(all_anomaly_tables)
        st.dataframe(full_anomaly_df.sort_values(['store_nbr', 'date']))
    else:
        st.info("目前選擇範圍內未發現任何異常點。")

if __name__ == '__main__':
    main()
