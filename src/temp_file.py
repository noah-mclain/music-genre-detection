from .models import CNNLSTMAttentionModel, SimpleModel

model = CNNLSTMAttentionModel()
print(sum(p.numel() for p in model.parameters()))

model = SimpleModel()
print(sum(p.numel() for p in model.parameters()))
