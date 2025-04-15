import pandas as pd
import zipfile
import requests
import io
import nltk
import streamlit as st
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import en_core_web_md
nlp = en_core_web_md.load()




# ----- Funções de Processamento e Modelagem -----
@st.cache_data
def carregar_dados():
    """
    Baixa o dataset de tweets sobre companhias aéreas.
    O dataset é extraído de um arquivo zip e carregado em um DataFrame.
    """
    url = 'https://github.com/alexvaroz/data_science_alem_do_basico/raw/refs/heads/master/tweets_airlines.zip'
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
    df = pd.read_csv(z.open(csv_filename))
    return df


def analisar_sentimento_textblob(text):
    """
    Analisa o sentimento de um tweet usando o TextBlob.
    - 'positivo' se o sentimento for positivo,
    - 'negativo' se o sentimento for negativo,
    - 'neutro' se o sentimento não for forte.
    """
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'


def treinar_tfidf_model(df):
    """
    Treina um modelo de aprendizado de máquina usando TF-IDF e Regressão Logística.
    O TF-IDF transforma o texto em números para que o modelo possa entender.
    A Regressão Logística é usada para fazer a previsão do sentimento do tweet.
    """
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['airline_sentiment'], test_size=0.2,
                                                        random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return y_test, y_pred, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)


def treinar_spacy_model(df):
    """
    Treina um modelo de aprendizado de máquina usando embeddings do SpaCy e Regressão Logística.
    O SpaCy transforma o texto em vetores numéricos (embeddings) e a Regressão Logística é usada para prever o sentimento.
    """
    df = df.copy()

    def get_vector(text):
        doc = nlp(str(text))
        return doc.vector if doc.has_vector else np.zeros(nlp.vocab.vectors_length)

    df['spacy_vector'] = df['text'].apply(get_vector)

    X = np.vstack(df['spacy_vector'].values)
    y = df['airline_sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred, accuracy_score(y_test, y_pred), classification_report(y_test, y_pred)


# ----- Interface do Streamlit -----
st.set_page_config(page_title="Análise de Sentimentos Comparativa", layout="wide")
st.title("✈️ Análise de Sentimentos com Comparação de Modelos")


# Carregar os dados
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

# Explicação sobre o método TextBlob
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

with st.expander("📋 Relatório TextBlob"):
    st.text(classification_report(df['airline_sentiment'], df['pred_textblob']))

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

y_test_tfidf, y_pred_tfidf, acc_tfidf, report_tfidf = treinar_tfidf_model(df)
st.metric("🎯 Acurácia (TF-IDF)", f"{acc_tfidf * 100:.2f}%")
with st.expander("📋 Relatório TF-IDF"):
    st.text(report_tfidf)

# --- SpaCy Embeddings + Logistic Regression ---
st.header("🧠 SpaCy Embeddings + Logistic Regression")

# Explicação sobre o método SpaCy + Logistic Regression
with st.expander("ℹ️ O que é SpaCy + Logistic Regression?"):
    st.markdown("""
    O **SpaCy** é uma biblioteca que cria representações vetoriais chamadas **embeddings**. Cada palavra é transformada em um vetor de números que captura o significado semântico da palavra.
    Com isso, a **Regressão Logística** usa esses vetores para classificar o sentimento de cada tweet.

    - **Embeddings do SpaCy**: Captura o significado semântico das palavras.
    - **Regressão Logística**: Classifica o sentimento com base nos embeddings.
    """)

y_test_spacy, y_pred_spacy, acc_spacy, report_spacy = treinar_spacy_model(df)
st.metric("🎯 Acurácia (SpaCy)", f"{acc_spacy * 100:.2f}%")
with st.expander("📋 Relatório SpaCy"):
    st.text(report_spacy)

# Comparação visual (gráfico de linha)
st.header("📈 Comparação de Acurácia entre os modelos")

# Gráfico de linha comparando os três modelos
acuracia_df = pd.DataFrame({
    "Modelos": ["TextBlob", "TF-IDF + LR", "SpaCy + LR"],
    "Acurácia": [acc_blob, acc_tfidf, acc_spacy]
})

st.line_chart(acuracia_df.set_index("Modelos"))

# --- Conclusão ---
st.header("🔚 Conclusão")

st.markdown("""
Com base nos resultados obtidos, podemos comparar o desempenho de cada modelo utilizado para análise de sentimentos:

1. **TextBlob**:
   - Acurácia: **46.44%**
   - O modelo **TextBlob** obteve uma acurácia relativamente baixa, com desempenho fraco, especialmente em identificar sentimentos **negativos** e **positivos**. Isso ocorre porque o TextBlob é uma abordagem mais simples que utiliza análise de polaridade, mas que não leva em conta o contexto mais profundo das palavras.

2. **TF-IDF + Logistic Regression**:
   - Acurácia: **79.58%**
   - O modelo **TF-IDF + Regressão Logística** obteve a melhor acurácia entre os três. Ele usou a técnica de transformação **TF-IDF**, que cria representações numéricas das palavras, permitindo que o modelo compreenda melhor o contexto semântico do texto. O classificador de **Regressão Logística** fez um trabalho eficaz de identificar os sentimentos presentes.

3. **SpaCy Embeddings + Logistic Regression**:
   - Acurácia: **74.59%**
   - O modelo **SpaCy + Regressão Logística** obteve uma boa acurácia, mas ficou atrás do TF-IDF. Embora o SpaCy utilize **embeddings**, que capturam o significado semântico das palavras de forma mais profunda, o modelo não teve um desempenho tão superior quanto o TF-IDF, possivelmente por causa de como o SpaCy lida com as representações vetoriais.

### Conclusão Final:
**TF-IDF + Regressão Logística** foi o modelo que apresentou o melhor desempenho, com a maior acurácia de **79.58%**. Esse modelo é eficaz em transformar o texto em uma representação numérica que captura palavras relevantes para a tarefa de análise de sentimentos. Apesar de **SpaCy** fornecer uma representação semântica mais profunda das palavras, o TF-IDF mostrou ser mais eficiente para esta tarefa específica.
""")
