import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

# 1. Загрузка данных
path = kagglehub.dataset_download("subhajournal/iotintrusion")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Опционально: если датасет всё равно слишком большой для вашего ПК,
# раскомментируйте строку ниже, чтобы взять случайные 50 000 строк для теста
# df = df.sample(n=50000, random_state=42)

df = df.dropna()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Оптимизированное кодирование категориальных признаков
# Вместо get_dummies применяем LabelEncoder ко всем колонкам типа object
for col in X.select_dtypes(include=['object']).columns:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 3. Разделение и масштабирование
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Использование быстрой версии градиентного бустинга
model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)

print("Начало обучения HistGradientBoostingClassifier (это займет немного времени)...")
model.fit(X_train, y_train)

# 5. Оценка
y_pred = model.predict(X_test)

print("\nLLM Model 1: Hist Gradient Boosting")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le_target.classes_))