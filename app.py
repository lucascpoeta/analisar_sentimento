import pandas as pd
import zipfile
import requests
import io
import nltk
import streamlit as st
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# ----- Funções de Processamento e Modelagem -----
@st.cache_data
def carregar_dados():
    url = 'https://github.com/alexvaroz/data_science_alem_do_basico/raw/refs/heads/master/tweets_airlines.zip'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
    df = pd.read_csv(z.open(csv_filename))
    return df


def analisar_sentimento_textblob(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def treinar_tfidf_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['airline_sentiment'], test_size=0.2,
                                                        random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
    return y_test, y_pred, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True), cm


# ----- Interface do Streamlit -----
st.set_page_config(page_title="Análise de Sentimentos Comparativa", layout="wide")
st.title("✈️ Análise de Sentimentos com Comparação de Modelos")

with st.expander("Introdução"):
    st.markdown("""
   A análise de sentimentos é uma técnica poderosa utilizada para extrair opiniões e emoções de textos, permitindo que as empresas compreendam melhor o feedback dos usuários e ajustem suas estratégias. Neste projeto, comparamos duas abordagens distintas para realizar a análise de sentimentos em tweets de companhias aéreas:

TextBlob: Uma abordagem simples e intuitiva que utiliza a polaridade do texto para classificar os sentimentos como positivos, negativos ou neutros.

TF-IDF + Regressão Logística: Uma abordagem mais robusta, que utiliza o TF-IDF (Term Frequency-Inverse Document Frequency) para extrair características do texto e um modelo de regressão logística para classificar os sentimentos.

Através dessa comparação, nosso objetivo é identificar qual modelo é mais eficaz ao analisar sentimentos em tweets de companhias aéreas. Para tanto, realizamos uma avaliação detalhada de cada técnica, apresentando as métricas de desempenho e resultados de acurácia, precisão, recall e F1-score, além de uma análise mais aprofundada com gráficos e tabelas.

Com isso, buscamos não apenas comparar os modelos, mas também entender suas limitações e potenciais, fornecendo insights valiosos sobre como escolher a melhor abordagem para a análise de sentimentos em diferentes contextos.
    """)


df = carregar_dados()
st.write("🔍 Amostra dos dados:")



st.dataframe(df[['text', 'airline_sentiment']].sample(5))

# Explicação geral das métricas
with st.expander("ℹ️Métricas de Avaliação?"):
    st.markdown("""
    **Precision (Precisão)**: Mede a acurácia das previsões feitas como **positivas**.
    - Quanto o modelo está correto quando prevê algo como positivo.

    **Recall (Revocação ou Sensibilidade)**: Mede a capacidade do modelo de identificar **todos** os positivos reais.
    - Quanto o modelo consegue identificar todos os casos positivos de fato.

    **F1-Score**: Combina a precisão e o recall em uma única métrica, equilibrando ambos.
    - Uma média ponderada entre a precisão e o recall. Uma boa métrica quando temos desequilíbrio entre as classes.

    **Acurácia**: Percentual de acertos totais (todas as previsões corretas divididas pelo total de previsões).
    - A proporção de previsões corretas feitas pelo modelo.

    **Macro avg**: Média das métricas (precisão, recall e F1) calculadas para cada classe, sem levar em consideração o número de exemplos em cada classe.

    **Weighted avg**: Média ponderada das métricas, considerando a proporção de cada classe no total de dados.

    Essas métricas ajudam a entender **onde o modelo acerta mais** e **onde pode melhorar**. Elas são importantes para avaliar a qualidade da previsão do modelo, especialmente quando as classes podem estar desequilibradas.
    """)

# --- TextBlob ---
st.header("🔠 TextBlob")

with st.expander("ℹ️ O que é o TextBlob?"):
    st.markdown("""
    O **TextBlob** é uma biblioteca simples e poderosa para análise de sentimentos. Ela atribui um valor de **polaridade** a cada texto. 
    - **Polaridade positiva**: Sentimentos positivos.
    - **Polaridade negativa**: Sentimentos negativos.
    - **Polaridade neutra**: Sentimentos neutros.

    Ele calcula a polaridade do texto e, com isso, conseguimos identificar rapidamente se o sentimento é positivo, negativo ou neutro.
    """)

df['pred_textblob'] = df['text'].apply(analisar_sentimento_textblob)
acc_blob = accuracy_score(df['airline_sentiment'], df['pred_textblob'])
st.metric("🎯 Acurácia (TextBlob)", f"{acc_blob * 100:.2f}%")

cm_blob = confusion_matrix(df['airline_sentiment'], df['pred_textblob'], labels=['positive', 'negative', 'neutral'])
acertos_por_classe_blob = {
    "positive": int(cm_blob[0, 0]),
    "negative": int(cm_blob[1, 1]),
    "neutral": int(cm_blob[2, 2])
}
erros_por_classe_blob = {
    "positive": int(cm_blob[0, 1] + cm_blob[0, 2]),
    "negative": int(cm_blob[1, 0] + cm_blob[1, 2]),
    "neutral": int(cm_blob[2, 0] + cm_blob[2, 1])
}

option = st.selectbox("Escolha o gráfico ou tabela", ["Tabela", "Pie Chart", "Bar Chart"])

cores_distintas = ["#0000ff", "#a1caf1", "#89a3dc"]

if option == "Tabela":
    st.subheader("Acertos e Erros por Classe (TextBlob):")
    dados_tabela = pd.DataFrame({
        "Classe": ["Positive", "Negative", "Neutral"],
        "Acertos✅": [acertos_por_classe_blob["positive"], acertos_por_classe_blob["negative"],
                    acertos_por_classe_blob["neutral"]],
        "Erros❌": [erros_por_classe_blob["positive"], erros_por_classe_blob["negative"],
                  erros_por_classe_blob["neutral"]]
    })
    st.write(dados_tabela)

elif option == "Pie Chart":
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [acertos_por_classe_blob["positive"], acertos_por_classe_blob["negative"],
             acertos_por_classe_blob["neutral"]]

    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.4,
                                     marker=dict(colors=cores_distintas))])
    fig_pie.update_layout(title="Distribuição de Acertos por Classe (TextBlob)", margin=dict(t=20, b=20, l=20, r=20),
                          font=dict(size=10))
    st.plotly_chart(fig_pie, use_container_width=True)

elif option == "Bar Chart":
    fig_bar = go.Figure([go.Bar(x=list(acertos_por_classe_blob.keys()), y=list(acertos_por_classe_blob.values()),
                                marker=dict(color=cores_distintas))])
    fig_bar.update_layout(title="Acertos por Classe (TextBlob)", xaxis_title="Classe", yaxis_title="Número de Acertos",
                          font=dict(size=10))
    st.plotly_chart(fig_bar, use_container_width=True)

# Relatório como tabela (TextBlob)
with st.expander("📋 Relatório TextBlob"):
    report_blob = classification_report(df['airline_sentiment'], df['pred_textblob'], output_dict=True)
    df_report_blob = pd.DataFrame(report_blob).transpose()
    st.dataframe(df_report_blob.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

with st.expander("📋 Interpretação dos Resultados"):
    st.markdown("""

#### Classe **Positive**:
- O modelo acertou **1807** tweets classificados como positivos e errou **556**. A quantidade de acertos nesta classe sugere que o modelo tem uma boa performance para identificar sentimentos positivos, mas ainda com um número significativo de erros.

#### Classe **Negative**:
- Para sentimentos negativos, o modelo acertou **3235** tweets, mas errou **5943**. Esse número de erros é maior em comparação com as outras classes, o que indica que o modelo tem mais dificuldade em classificar corretamente tweets negativos. Isso pode sugerir a necessidade de ajustes no modelo ou mais dados de treinamento para melhorar a precisão em sentimentos negativos.

#### Classe **Neutral**:
- O modelo acertou **1757** tweets neutros e errou **1342**. A quantidade de acertos nesta classe é razoável, mas o número de erros também é significativo. Isso indica que o modelo pode ter dificuldade em distinguir sentimentos neutros dos outros tipos de sentimentos (positivos ou negativos).

### O que isso nos diz?
- **Desbalanceamento de Dados**: Se o número de erros for muito maior em uma classe específica, isso pode ser um indicativo de desbalanceamento nas classes ou de que o modelo não está conseguindo capturar bem as características dessa classe. O número de erros altos para a classe **Negative**, por exemplo, sugere que o modelo pode estar mais propenso a classificar erroneamente tweets negativos como neutros ou até mesmo positivos.

- **Ajustes no Modelo**: As classes **Negative** e **Neutral** parecem ter uma quantidade maior de erros em comparação com a classe **Positive**. Isso pode indicar que o modelo tem uma tendência a acertar mais os sentimentos positivos, mas tem dificuldades para discernir sentimentos negativos ou neutros. Talvez seja necessário ajustar o treinamento, com mais dados ou técnicas de balanceamento de classes.

- **Métricas Importantes**: O número de acertos e erros também afeta as métricas como **Precisão**, **Recall** e **F1-Score**, que são essenciais para medir o desempenho de um modelo em cenários com classes desbalanceadas. A precisão pode ser impactada negativamente se houver muitos erros, especialmente nas classes com menor representação, enquanto o recall pode ser baixo para as classes com maior número de erros.

### Possíveis Melhorias:
1. **Ajuste no modelo**: Melhorar o modelo de classificação, considerando ajustes de parâmetros ou técnicas como **SMOTE (Synthetic Minority Over-sampling Technique)** para balancear os dados de treinamento.
2. **Análise de erros**: Analisar os erros cometidos para entender se há padrões nos tweets que são classificados incorretamente, como a presença de palavras ambíguas ou ironia que podem afetar a classificação.
3. **Mais dados de treinamento**: Coletar mais dados, especialmente para as classes com menos acertos (como **Negative**), pode ajudar a melhorar o desempenho do modelo.
 """)

# --- TF-IDF + Logistic Regression ---
st.header("📊 TF-IDF + Logistic Regression")

# Explicação sobre o método TF-IDF + Logistic Regression
with st.expander("ℹ️ O que é TF-IDF + Logistic Regression?"):
    st.markdown("""
    O **TF-IDF (Term Frequency-Inverse Document Frequency)** é uma técnica que transforma o texto em números, dando mais peso a palavras importantes.
    Em seguida, o modelo de **Regressão Logística** usa esses números para fazer previsões. Ele é um classificador, ou seja, ele tenta adivinhar a categoria (positivo/negativo) de um tweet com base nas palavras.

    - **TF-IDF**: Transforma o texto em números.
    - **Regressão Logística**: Classifica o sentimento com base nos números gerados pelo TF-IDF.
    """)


y_test_tfidf, y_pred_tfidf, acc_tfidf, report_tfidf_dict, cm_tfidf = treinar_tfidf_model(df)
st.metric("🎯 Acurácia (TF-IDF)", f"{acc_tfidf * 100:.2f}%")

acertos_por_classe_tfidf = {
    "positive": int(cm_tfidf[0, 0]),
    "negative": int(cm_tfidf[1, 1]),
    "neutral": int(cm_tfidf[2, 2])
}
erros_por_classe_tfidf = {
    "positive": int(cm_tfidf[0, 1] + cm_tfidf[0, 2]),
    "negative": int(cm_tfidf[1, 0] + cm_tfidf[1, 2]),
    "neutral": int(cm_tfidf[2, 0] + cm_tfidf[2, 1])
}

option_tfidf = st.selectbox("Escolha o gráfico ou tabela para TF-IDF", ["Tabela", "Pie Chart", "Bar Chart"])

if option_tfidf == "Tabela":
    st.subheader("Acertos e Erros por Classe (TF-IDF):")
    dados_tabela_tfidf = pd.DataFrame({
        "Classe": ["Positive", "Negative", "Neutral"],
        "Acertos✅": [acertos_por_classe_tfidf["positive"], acertos_por_classe_tfidf["negative"],
                    acertos_por_classe_tfidf["neutral"]],
        "Erros❌": [erros_por_classe_tfidf["positive"], erros_por_classe_tfidf["negative"],
                  erros_por_classe_tfidf["neutral"]]
    })
    st.write(dados_tabela_tfidf)

elif option_tfidf == "Pie Chart":
    sizes_tfidf = [acertos_por_classe_tfidf["positive"], acertos_por_classe_tfidf["negative"],
                   acertos_por_classe_tfidf["neutral"]]

    fig_pie_tfidf = go.Figure(data=[go.Pie(labels=labels, values=sizes_tfidf, hole=0.4,
                                           marker=dict(colors=cores_distintas))])
    fig_pie_tfidf.update_layout(title="Distribuição de Acertos por Classe (TF-IDF)",
                                margin=dict(t=20, b=20, l=20, r=20), font=dict(size=10))
    st.plotly_chart(fig_pie_tfidf, use_container_width=True)

elif option_tfidf == "Bar Chart":
    fig_bar_tfidf = go.Figure(
        [go.Bar(x=list(acertos_por_classe_tfidf.keys()), y=list(acertos_por_classe_tfidf.values()),
                marker=dict(color=cores_distintas))])
    fig_bar_tfidf.update_layout(title="Acertos por Classe (TF-IDF)", xaxis_title="Classe",
                                yaxis_title="Número de Acertos", font=dict(size=10))
    st.plotly_chart(fig_bar_tfidf, use_container_width=True)

# Relatório como tabela (TF-IDF)
with st.expander("📋 Relatório TF-IDF"):
    df_report_tfidf = pd.DataFrame(report_tfidf_dict).transpose()
    st.dataframe(df_report_tfidf.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))
with st.expander("📋 Interpretação dos Resultados"):
    st.markdown("""


#### Classe **Positive**:
- O modelo acertou **289** tweets classificados como positivos e errou **170**. Embora o número de acertos seja significativo, a quantidade de erros também é alta, indicando que o modelo pode ter dificuldade em classificar corretamente alguns sentimentos positivos. Essa discrepância pode ocorrer devido à presença de palavras ambíguas ou a complexidade do modelo.

#### Classe **Negative**:
- Para sentimentos negativos, o modelo acertou **1758** tweets, mas errou apenas **131**. O modelo parece ter um bom desempenho em classificar sentimentos negativos, com um número muito baixo de erros. Isso sugere que a característica dos tweets negativos está bem representada no treinamento e que o modelo consegue capturar adequadamente os sentimentos negativos.

#### Classe **Neutral**:
- O modelo acertou **283** tweets neutros e errou **297**. Apesar de um número considerável de erros, o modelo acertou uma quantidade razoável de tweets neutros. No entanto, os erros nesta classe indicam que o modelo pode estar confundindo tweets neutros com os sentimentos positivos ou negativos, o que é comum em modelos que analisam textos curtos.

### O que isso nos diz?
- **Desbalanceamento de Dados**: O modelo parece ter uma boa performance para a classe **Negative**, mas com dificuldades para identificar corretamente os sentimentos **Positive** e **Neutral**. Isso pode ser um indicativo de desbalanceamento no número de exemplos dessas classes no conjunto de treinamento ou de uma complexidade maior ao tentar distinguir os sentimentos neutros e positivos.

- **Ajustes no Modelo**: A classe **Neutral** apresenta um número relativamente alto de erros. Isso sugere que o modelo pode ser aprimorado com mais dados ou por meio de técnicas de balanceamento, como **SMOTE**, para melhorar a detecção de sentimentos neutros.

- **Métricas Importantes**: A precisão em sentimentos **Negative** parece ser muito boa, enquanto a acurácia de **Positive** e **Neutral** pode ser melhorada. Isso pode impactar diretamente métricas como **F1-Score**, **Recall** e **Precisão**, que precisam ser avaliadas para entender melhor a eficácia do modelo em diferentes classes.

### Possíveis Melhorias:
1. **Reforçar a classificação para Neutral**: Uma solução pode ser o uso de **técnicas de aumento de dados** (data augmentation) ou a reavaliação dos exemplos de treinamento da classe neutra, que podem ter características semelhantes a outras classes.
2. **Ajuste nos parâmetros do TF-IDF**: A utilização de parâmetros como **max_features** e **ngram_range** pode melhorar a captura de características específicas de cada classe.
3. **Balanceamento de Dados**: Técnicas como **SMOTE** ou **undersampling** para as classes desbalanceadas podem ser aplicadas para melhorar a performance nas classes com maior número de erros.
 """)

import plotly.graph_objects as go

# Comparação de acurácia
st.header("📈 Comparação de Acurácia entre os modelos")

# Criar DataFrame para comparação de acurácia
acuracia_df = pd.DataFrame({
    "Modelos": ["TextBlob", "TF-IDF + LR"],
    "Acurácia": [acc_blob, acc_tfidf]
})

# Criar gráfico com Plotly
fig = go.Figure()

# Adicionar a linha para a acurácia
fig.add_trace(go.Scatter(
    x=acuracia_df["Modelos"],
    y=acuracia_df["Acurácia"],
    mode='lines+markers',
    name='Acurácia',
    marker=dict(color='blue', size=10),  # Personalizando o marcador
    line=dict(color='blue', width=2)     # Personalizando a linha
))

# Personalizar layout do gráfico
fig.update_layout(
    xaxis_title="Modelos",
    yaxis_title="Acurácia (%)",
    yaxis=dict(range=[0, 1]),  # Definir intervalo do eixo Y de 0 a 1 (ou de 0% a 100%)
    template="plotly_dark",  # Tema para o gráfico
    font=dict(size=12),
    showlegend=False  # Remover legenda
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig)


# Conclusão
st.header("🔚 Conclusão")
st.markdown(f"""
Com base nos resultados da análise de desempenho dos modelos de TextBlob e TF-IDF + Regressão Logística, podemos observar uma diferença significativa na eficácia de cada abordagem.

TextBlob obteve uma acurácia de 46.44%, o que indica um desempenho relativamente baixo em relação à tarefa de análise de sentimentos. Embora o TextBlob seja uma ferramenta simples e fácil de usar para análise de sentimentos, seu desempenho pode ser afetado por sua abordagem mais básica, que não considera de forma tão aprofundada as nuances do texto.

Por outro lado, o modelo TF-IDF + Regressão Logística alcançou uma acurácia de 79.58%, um desempenho muito superior. O TF-IDF é uma técnica que captura a importância relativa das palavras em um documento, enquanto a Regressão Logística é um classificador robusto, especialmente quando combinado com representações textuais como o TF-IDF. Essa combinação permite ao modelo capturar melhor as relações semânticas entre as palavras e suas contribuições para a classificação de sentimentos.

Conclusão Final: A análise sugere que o modelo TF-IDF + Regressão Logística é o mais eficaz para a tarefa de análise de sentimentos, apresentando uma acurácia significativamente maior que o TextBlob. Isso destaca a importância de técnicas mais sofisticadas de representação textual e de aprendizado de máquina, como o TF-IDF, para melhorar a precisão das previsões em tarefas de classificação de sentimentos.
""")
