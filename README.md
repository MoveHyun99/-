# -
학습과 과제를 다루는

## CNN 예제

`cnn_model.py`에는 CIFAR-10과 같은 소규모 이미지 분류 데이터셋을 위한 간단한 합성곱 신경망과 학습 루프가 포함되어 있습니다. PyTorch와 Torchvision만으로 동작하며, 다음과 같이 사용할 수 있습니다.

```
python cnn_model.py
```

또는 모듈을 가져와 사용자 정의 설정으로 실행할 수도 있습니다.

```
from cnn_model import SimpleCNN, TrainingConfig, build_cifar10_loaders, train

config = TrainingConfig(epochs=5, batch_size=128, learning_rate=3e-4)
train_loader, val_loader = build_cifar10_loaders("./data", batch_size=config.batch_size)
model = SimpleCNN(num_classes=10)
history = train(model, train_loader, val_loader, config)
print(history)
```
