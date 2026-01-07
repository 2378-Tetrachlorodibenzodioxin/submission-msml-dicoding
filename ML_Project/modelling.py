import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model():
    # 1. Aktifkan Autologging
    # Ini akan otomatis mencatat parameter, metrik, dan model ke MLflow
    mlflow.sklearn.autolog()

    # Set nama eksperimen agar rapi di dashboard
    # mlflow.set_experiment("Eksperimen_Penguins_Basic")

    # Mulai Run MLflow
    with mlflow.start_run():
        print("Memuat dataset...")
        # Load data yang sudah bersih (angka semua)
        df = pd.read_csv('penguins_clean.csv')

        # Pisahkan Fitur dan Target
        X = df.drop(columns=['species'])
        y = df['species']

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Melatih model Random Forest...")
        # Inisialisasi Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Training Model
        # Saat .fit() dipanggil, MLflow otomatis mencatat semuanya karena autolog() aktif
        model.fit(X_train, y_train)

        # Evaluasi
        acc = model.score(X_test, y_test)
        print(f"Training selesai. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_model()