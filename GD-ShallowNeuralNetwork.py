import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Carregar o dataset
dataset = pd.read_csv('C:/Users/guite/OneDrive/Documentos/Faculdade/6o Periodo/DL - Pedro/Projeto DeepLearning/datasets/results.csv', encoding='latin-1')

# Remover colunas desnecessárias
dataset = dataset.drop(columns=['neutral', 'date', 'city'])

# Criar colunas indicando se a partida foi disputada no país do time da casa ou visitante
dataset['home_team_country'] = dataset['home_team'] == dataset['country']
dataset['away_team_country'] = dataset['away_team'] == dataset['country']

# Codificar a coluna 'tournament' para indicar se é amistoso ou competição
dataset['is_friendly'] = dataset['tournament'] == 'Friendly'

# Transformar as colunas booleanas em inteiros (0 ou 1)
dataset['home_team_country'] = dataset['home_team_country'].astype(int)
dataset['away_team_country'] = dataset['away_team_country'].astype(int)
dataset['is_friendly'] = dataset['is_friendly'].astype(int)

# Criar a coluna de resultado com base na diferença de gols
def get_result(row):
    goal_difference = row['home_score'] - row['away_score']
    return goal_difference

dataset['goal_difference'] = dataset.apply(get_result, axis=1)

# Selecionar as colunas que precisam de One-Hot Encoding
categorical_columns = ['home_team', 'away_team', 'tournament', 'country']

# Aplicar One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(dataset[categorical_columns])

# Criar um DataFrame com as novas colunas codificadas
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenar as colunas codificadas com o dataset original (excluindo as colunas categóricas originais)
dataset = pd.concat([dataset.drop(columns=categorical_columns), encoded_df], axis=1)

# Remover colunas de gols
dataset = dataset.drop(columns=['home_score', 'away_score'])

# Separar as features (X) do target (y)
X = dataset.drop(columns=['goal_difference'])
y = dataset['goal_difference']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

# Aplicar o StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Arquitetura da rede neural Shallow Net para regressão
model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='linear'))

# Compilar o modelo
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
print(model.summary())

# Treinamento
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Avaliação
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f'Mean Absolute Error: {mae:.2f}')

# Predições no conjunto de teste
y_pred = model.predict(X_test_scaled)