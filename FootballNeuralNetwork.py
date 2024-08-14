import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

# Treinar o modelo
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Fazer previsões
y_pred = rf.predict(X_test)

# Avaliar o modelo
print(rf.score(X_test, y_test))
print(classification_report(y_test, y_pred))