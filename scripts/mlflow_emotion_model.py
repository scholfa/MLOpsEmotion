import mlflow.pyfunc
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class EmotionRecognitionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.artifacts["model_dir"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, context, model_input):
        inputs = self.feature_extractor(
            model_input,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        return [self.id2label[p.item()] for p in preds]
