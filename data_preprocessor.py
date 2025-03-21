import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def clean_data(self):
        # Удаление строк с пропущенными значениями
        self.data = self.data.dropna()

    def encode_categorical_features(self):
        # Кодируем категориальные признаки (island и sex)
        self.data['island'] = LabelEncoder().fit_transform(self.data['island'])
        self.data['sex'] = LabelEncoder().fit_transform(self.data['sex'])

    def split_data(self, target_column='species', test_size=0.2, random_state=None):
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        # Кодирование целевой переменной
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def normalize_data(self):
        # Нормализация
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def get_processed_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test