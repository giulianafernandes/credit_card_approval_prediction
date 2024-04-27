import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# Classes para a Pipeline

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop = ['id_cliente']):
        self.feature_to_drop = feature_to_drop
    def fit(self, df):
        return self
    def transform(self, df):
        if(set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print('Uma ou mais features não estão no Dataframe - drop_features')
            return df
        
class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler = ['tam_familia', 'rendimento_anual', 'idade', 'anos_empregado']):
        self.min_max_scaler = min_max_scaler
    def fit(self, df):
        return self
    def transform(self, df):
        if (set(self.min_max_scaler).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler] = min_max_enc.fit_transform(df[self.min_max_scaler])
            return df
        else:
            print('Uma ou mais features não estão no Dataframe - min_max')
            return df
        
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encoding = ['categoria_renda', 'estado_civil', 'moradia', 'ocupacao']):
        self.one_hot_encoding = one_hot_encoding
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if(set(self.one_hot_encoding).issubset(df.columns)):
            # função para rodar o OneHotEncoder nas features
            def onehot_features(df, one_hot_encoding):
                onehot_features = OneHotEncoder()
                onehot_features.fit(df[one_hot_encoding])
                
                #obtendo o nome das colunas
                feature_names = onehot_features.get_feature_names_out(one_hot_encoding)
                
                # transformando o array em um dataframe com o nome das colunas
                df = pd.DataFrame(onehot_features.transform(df[self.one_hot_encoding]).toarray(),
                                  columns= feature_names, index=df.index)
                
                return df
            
            #função para concatenar as features que passaram pelo fit dos dados com o restante 
            def concat_df(df, onehotenc_df, one_hot_encoding):
                outras_features = [feature for feature in df.columns if feature not in one_hot_encoding]
                df_concat = pd.concat([onehotenc_df, df[outras_features]], axis=1)
                return df_concat
            
            df_onehotencoding = onehot_features(df, self.one_hot_encoding)
            
            # retorna o df concatenado
            df_full = concat_df(df, df_onehotencoding, self.one_hot_encoding)
            return df_full
        
        else:
            print('Uma ou mais features não estão no Dataframe - min_max')
            return df
        
class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature = ['grau_escolaridade']):
        self.ordinal_feature = ordinal_feature
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'grau_escolaridade' in df.columns:
            ord_encoder = OrdinalEncoder()
            df[self.ordinal_feature] = ord_encoder.fit_transform(df[self.ordinal_feature])
            return df
        else:
            print('grau_escolaridade não está no Dataframe')
            return df
        

class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'mau' in df.columns:
            # usando a função SMOTE para superamostrar a clase minoritária e corrigir dados desbalanceados
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns !='mau'], df['mau'])
            df_balanceado = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return df_balanceado
        else:
            print("O target não está no Dataframe")
            return df
        