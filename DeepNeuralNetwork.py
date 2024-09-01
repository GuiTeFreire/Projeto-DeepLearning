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

# Criar a coluna de resultado
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 0  # Vitória do time da casa
    elif row['home_score'] < row['away_score']:
        return 2  # Vitória do time visitante
    else:
        return 1  # Empate

dataset['result'] = dataset.apply(get_result, axis=1)

# Selecionar as colunas que precisam de One-Hot Encoding
categorical_columns = ['home_team', 'away_team', 'tournament', 'country']

# Aplicar One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(dataset[categorical_columns])

# Criar um DataFrame com as novas colunas codificadas
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenar as colunas codificadas com o dataset original (excluindo as colunas categóricas originais)
dataset = pd.concat([dataset.drop(columns=categorical_columns), encoded_df], axis=1)

# Separar as features (X) do target (y)
X = dataset.drop(columns=['result'])
y = dataset['result']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.3)

# Aplicar SMOTE no conjunto de treinamento
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Aplicar o StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# Arquitetura da rede neural Deep Net
model = keras.Sequential()

# Primeira camada densa com ReLU como função de ativação
model.add(keras.layers.Dense(64, input_shape=(1088,), activation='relu'))
model.add(keras.layers.BatchNormalization())

# Segunda camada densa (camada oculta)
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())

# Terceira camada densa (camada oculta)
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())

# Quarta camada densa (camada oculta)
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

# Camada de saída com 3 categorias e softmax para classificação multiclasse
model.add(keras.layers.Dense(3, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Treinamento
class_weights = {0: 1., 1: 2.3, 2: 1.2}
model.fit(X_train_scaled, y_train_sm, epochs=50, batch_size=32,
          validation_data=(X_test_scaled, y_test),
          class_weight=class_weights)

# Avaliação
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia: {accuracy*100:.2f}%')
