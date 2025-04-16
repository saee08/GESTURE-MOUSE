import json
import joblib
from sklearn.ensemble import RandomForestClassifier
print("All imports successful!")

class GestureTraining:
    def __init__(self, model_path="model/gesture_model.pkl", data_path="data/gesture_data.json"):
        self.model_path = model_path
        self.data_path = data_path
    
    def collect_gesture_data(self, label, landmarks):
        with open(self.data_path, "a") as file:
            json.dump({"label": label, "landmarks": landmarks}, file)
            file.write("\n")
    
    def train_model(self):
        X, y = [], []
        with open(self.data_path, "r") as file:
            for line in file:
                data = json.loads(line)
                X.append(data["landmarks"])
                y.append(data["label"])
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        joblib.dump(model, self.model_path)

        if __name__ == "__main__":
         training = GestureTraining()
         training.train_model()
