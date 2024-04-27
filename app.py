# importando as bibliotecas
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMax
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

dados = pd.read_csv('df_clean.csv')

st.write('# Simulador de Avaliação de Crédito')

st.write('### Idade')
input_idade = float(st.slider('Selecione a sua idade', 18,100 ))

st.write('### Grau de escolaridade')
input_escolaridade = st.selectbox('Qual é o seu grau de escolaridade?', dados['grau_escolaridade'].unique())

st.write('### Estado Civil')
input_estado_civil = st.selectbox('Qual é o seu estado civil', dados['estado_civil'].unique())

st.write('### Família')
input_tam_familia = float(st.slider('Selecione quantos membros tem a sua família', 1,20))

st.write('### Carro próprio')
input_carro = st.radio('Você possui um automóvel?', ['Sim', 'Não'])
input_carro_dict = {'Sim': 1,
                    'Não': 0}
input_carro = input_carro_dict.get(input_carro)

st.write('### Casa própria')
input_casa = st.radio('Você possui um imóvel?', ['Sim', 'Não'])
input_casa_dict = {'Sim': 1,
                    'Não': 0}
input_casa = input_casa_dict.get(input_casa)

st.write('### Tipo de residência')
input_tipo_moradia = st.selectbox('Qual é o seu tipo de moradia?', dados['moradia'].unique())

st.write('### Categoria de renda')
input_cat_renda = st.selectbox('Qual é a sua categoria de renda?', dados['categoria_renda'].unique())

st.write('### Ocupação')
input_ocupação = st.selectbox('Qual é a sua ocupação?', dados['ocupacao'].unique())

st.write('### Experiência profissional')
input_experiencia = float(st.slider('Qual é o seu tempo de experiência profissional?', 0,40))

st.write('### Rendimentos')
input_rendimentos = float(st.number_input('Digite aqui o seu rendimento anual R$', 0))

st.write('### Telefone Corporativo')
input_tel_corp = st.radio('Você tem telefone corporativo?', ['Sim', 'Não'])
input_tel_corp_dict = {'Sim': 1,
                       'Não': 0}
input_tel_corp = input_tel_corp_dict.get(input_tel_corp)

st.write('### Telefone Fixo')
input_tel_fixo = st.radio('Você tem telefone fixo?', ['Sim', 'Não'])
input_tel_fixo_dict = {'Sim': 1,
                       'Não': 0}
input_tel_fixo = input_tel_fixo_dict.get(input_tel_fixo)

st.write('### E-mail')
input_email = st.radio('Você tem um e-mail?', ['Sim', 'Não'])
input_email_dict = {'Sim': 1,
                       'Não': 0}
input_email = input_email_dict.get(input_email)


#criando uma lista com os valores inseridos pelo usuário
novo_cliente = [0, 
                input_carro,
                input_casa,
                input_tel_corp,
                input_tel_fixo,
                input_email,
                input_tam_familia,
                input_rendimentos,
                input_idade,
                input_experiencia,
                input_cat_renda,
                input_escolaridade,
                input_estado_civil,
                input_tipo_moradia,
                input_ocupação,
                0
                ]


# Separando os dados em treino e teste 
def data_split(df, test_size):
    seed = 1561651
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_df, test_df = data_split(dados, 0.2)

# criando o novo cliente
cliente_predict_df = pd.DataFrame([novo_cliente], columns=test_df.columns)

# concatenando o novo cliente ao df de teste
test_novo_cliente = pd.concat([test_df, cliente_predict_df], ignore_index=True)

#Pipeline
def df_pipeline(df):
    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('MinMaxScaler', OneHotEncodingNames()),
        ('OneHotEncoding', OrdinalFeature()),
        ('OrdinalFeatures', MinMax()),
    ])
    
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

# aplicando a pipeline
test_novo_cliente = df_pipeline(test_novo_cliente)

#retirando a coluna target
cliente_pred = test_novo_cliente.drop(['mau'], axis=1)

if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success('## Crédito Aprovado!')
        st.balloons()
    else:
        st.error('## Crédito não aprovado.')