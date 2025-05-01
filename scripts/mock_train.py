import time
def simulate_training():
    time.sleep(1)
    with open("models/emotion_model.pkl", "w") as f:
        f.write("mock model")
if __name__ == "__main__":
    simulate_training()
