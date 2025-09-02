"""
Titanic: бинарная классификация (survived 0/1)
В одном скрипте:
  - подготовка данных (seaborn или Kaggle CSV)
  - MLP на Keras (TensorFlow)
  - MLP на PyTorch
  - сравнение Adam vs SGD, L2 и Dropout
Запуск из PyCharm: Run -> Run 'titanic_nn'
"""

import os
import numpy as np
import pandas as pd

# ===== (1) ЗАГРУЗКА И ПРЕПРОЦЕССИНГ ДАННЫХ =====

def load_titanic():
    """
    Пытаемся загрузить датасет:
      1) через seaborn.load_dataset("titanic") (нужен интернет)
      2) если не вышло — читаем CSV из ./data/titanic.csv (Kaggle-версия)

    Возвращаем:
      X_train, X_test : np.float32
      y_train, y_test : np.float32
      feature_names   : список имён признаков (после one-hot)
    """
    # 1) seaborn
    df = None
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # seaborn: колонки в нижнем регистре
        # оставим наиболее полезные и «чистые» столбцы
        cols = [
            "survived", "pclass", "sex", "age", "sibsp", "parch",
            "fare", "embarked", "alone", "who", "adult_male"
        ]
        df = df[cols].copy()
        source = "seaborn"
    except Exception:
        # 2) Kaggle CSV (обычно в верхнем регистре и с иными именами)
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "titanic.csv")
        csv_path = os.path.abspath(csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Не найден датасет. Либо включи интернет для seaborn, "
                f"либо положи Kaggle CSV в {csv_path}"
            )
        df = pd.read_csv(csv_path)
        source = "kaggle_csv"
        # Приведём Kaggle-колонки к «seaborn»-стилю
        # Kaggle: Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
        rename_map = {
            "Survived": "survived",
            "Pclass": "pclass",
            "Sex": "sex",
            "Age": "age",
            "SibSp": "sibsp",
            "Parch": "parch",
            "Fare": "fare",
            "Embarked": "embarked",
        }
        df = df.rename(columns=rename_map)
        # Добавим «who/alone/adult_male» в духе seaborn (прибл.)
        df["adult_male"] = ((df["sex"].str.lower() == "male") & (df["age"].fillna(0) >= 18)).astype(int)
        df["alone"] = ((df["sibsp"].fillna(0) + df["parch"].fillna(0)) == 0).astype(int)
        # who: man/woman/child
        who = np.where(df["age"].fillna(0) < 18, "child",
                       np.where(df["sex"].str.lower() == "male", "man", "woman"))
        df["who"] = who

        cols = [
            "survived", "pclass", "sex", "age", "sibsp", "parch",
            "fare", "embarked", "alone", "who", "adult_male"
        ]
        df = df[cols].copy()

    # Заполнение пропусков
    num_cols = ["age", "fare", "pclass", "sibsp", "parch"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    cat_cols = ["sex", "embarked", "alone", "who", "adult_male"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
        df[c] = df[c].cat.add_categories("Unknown").fillna("Unknown")

    # One-hot для категорий
    X = pd.get_dummies(df.drop("survived", axis=1), drop_first=True)
    y = df["survived"].astype(np.float32).values

    # Train/Test split + масштабирование (учим скейлер ТОЛЬКО на train)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"[INFO] Источник данных: {source}, X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32), list(X.columns)


# ===== (2) KERAS / TENSORFLOW =====

def run_keras(X_train, y_train, X_test, y_test, optimizer="adam", l2=1e-4, dropout=0.2,
              epochs=40, batch_size=32, seed=42):
    """
    MLP на Keras с L2 + Dropout. Сравниваем Adam и SGD.
    Возвращаем accuracy на тесте.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # Фиксируем сиды для воспроизводимости
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Строим модель
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(dropout),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid'),
    ])

    # Оптимизатор
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=1e-3)
    else:
        opt = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Тренировка (validation_split для мониторинга; verbose=0 — тихо)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Оценка на тесте
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return float(test_acc)


# ===== (3) PYTORCH =====

def run_torch(X_train, y_train, X_test, y_test, optimizer="adam", weight_decay=1e-4,
              dropout=0.2, epochs=40, batch_size=32, seed=42):
    """
    MLP на PyTorch. BCEWithLogitsLoss (то есть без sigmoid в самой модели).
    L2 = weight_decay, Dropout как обычно. Возвращаем accuracy на тесте.
    """
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Тензоры
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test,  dtype=torch.float32)
    yte = torch.tensor(y_test,  dtype=torch.float32)

    # Даталоадер
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    # Модель
    class MLP(nn.Module):
        def __init__(self, in_dim, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 16),    nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(16, 1)      # без Sigmoid — будем считать BCEWithLogitsLoss
            )
        def forward(self, x):
            return self.net(x).squeeze(1)

    model = MLP(X_train.shape[1], dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Оптимизатор
    if optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=weight_decay)

    # Тренировка
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    # Оценка
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).cpu().numpy().astype(int).ravel()
        truth  = yte.numpy().astype(int).ravel()
        acc    = (preds == truth).mean()
    return float(acc)


# ===== (4) СРАВНЕНИЕ И ЗАПУСК =====

def main():
    # Загружаем и готовим данные
    X_train, X_test, y_train, y_test, feat_names = load_titanic()

    # Запуски Keras
    acc_k_adam = run_keras(X_train, y_train, X_test, y_test,
                           optimizer="adam", l2=1e-4, dropout=0.2, epochs=40)
    acc_k_sgd  = run_keras(X_train, y_train, X_test, y_test,
                           optimizer="sgd", l2=1e-4, dropout=0.2, epochs=40)

    # Запуски Torch
    acc_t_adam = run_torch(X_train, y_train, X_test, y_test,
                           optimizer="adam", weight_decay=1e-4, dropout=0.2, epochs=40)
    acc_t_sgd  = run_torch(X_train, y_train, X_test, y_test,
                           optimizer="sgd", weight_decay=1e-4, dropout=0.2, epochs=40)

    # Табличка результатов
    res = pd.DataFrame([
        {"framework": "Keras", "optimizer": "Adam", "accuracy": acc_k_adam},
        {"framework": "Keras", "optimizer": "SGD",  "accuracy": acc_k_sgd},
        {"framework": "PyTorch", "optimizer": "Adam","accuracy": acc_t_adam},
        {"framework": "PyTorch", "optimizer": "SGD", "accuracy": acc_t_sgd},
    ])
    print("\n=== RESULTS (test accuracy) ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
