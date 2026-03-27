import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from utils import obter_dados_brent, treinar_modelo_lstm, gerar_previsao

def exibir_previsao():
    st.title("📈 Previsão do Preço do Brent com LSTM")

    # Obter dados históricos do Brent
    df = obter_dados_brent()
    data_max = df["Date"].max()  # Última data disponível no dataframe

    # Adicionando o campo para o número de dias de previsão
    num_dias = st.number_input(
        "Quantos dias deseja prever?",
        min_value=1,
        max_value=365,  # Limite máximo de dias
        value=30,  # Valor padrão
        step=1
    )

    # Adicionando o botão para gerar a previsão
    if st.button("Gerar Previsão"):
        st.write("🔄 Calculando previsão...")

        # Treinamento do modelo e geração da previsão com base no número de dias
        modelo, scaler = treinar_modelo_lstm(df)
        df_previsao = gerar_previsao(modelo, scaler, df, num_dias)

        # Criando o gráfico de previsões
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode='lines', name='Preço Histórico', line=dict(color='#6baed6')))
        fig.add_trace(go.Scatter(x=df_previsao["Data"], y=df_previsao["Preço Previsto"], mode='lines', name='Preço Previsto', line=dict(color='#ff7f0e', dash='dot')))

        fig.update_layout(title="Previsão do Preço do Brent", xaxis_title="Data", yaxis_title="Preço (USD)", template="plotly_dark")

        # Exibindo o gráfico no Streamlit
        st.plotly_chart(fig)

        # Exibindo a tabela com os dados da previsão
        st.subheader("📋 Dados da Previsão")
        st.dataframe(df_previsao)
    else:
        st.write("⏳ Clique no botão para gerar a previsão.")
