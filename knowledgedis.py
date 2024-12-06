import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet34, mobilenet_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = ImageFolder(root='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ImageFolder(root='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载教师模型
def load_teacher_model():
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(512, 37)
    model.load_state_dict(torch.load('teacher_model_state_dict.pth'))
    model = model.to(device)
    model.eval()
    return model

teacher_model = load_teacher_model()

# 定义简单 CNN 学生模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 112 * 112, 37)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

# 定义 MobileNetV2 学生模型
def load_mobilenet_student(num_classes=37):
    student_model = mobilenet_v2(pretrained=True)
    student_model.classifier[1] = nn.Linear(student_model.classifier[1].in_features, num_classes)
    return student_model.to(device)

# 定义蒸馏损失
def distillation_loss(y_student, y_teacher, labels, T=2.0, alpha=0.8):
    KL_loss = nn.KLDivLoss()(F.log_softmax(y_student / T, dim=1), F.softmax(y_teacher / T, dim=1)) * (T * T * alpha)
    CE_loss = nn.CrossEntropyLoss()(y_student, labels) * (1 - alpha)
    return KL_loss + CE_loss

# 训练学生模型
def train_student(teacher_model, student_model, train_loader, test_loader, optimizer, T, alpha, epochs=20, patience=5):
    """
    训练学生模型，包含 Early Stopping 功能。
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        student_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            loss = distillation_loss(student_outputs, teacher_outputs, labels, T, alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证阶段
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student_model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # 检查是否需要更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(student_model.state_dict(), 'best_student_model.pth')  # 保存最佳模型
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        # 触发 Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    student_model.load_state_dict(torch.load('best_student_model.pth'))
    print("Loaded best model from checkpoint.")


# 评估模型
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 测试不同 alpha 和 T 值
def test_different_parameters(teacher_model, train_loader, test_loader, alpha_values, temperature_values):
    results = {"SimpleCNN": [], "MobileNetV2": []}
    for alpha in alpha_values:
        for T in temperature_values:
            for model_name in results.keys():
                print(f"\nTesting {model_name} with alpha={alpha}, T={T}")
                if model_name == "SimpleCNN":
                    student_model = SimpleCNN().to(device)
                elif model_name == "MobileNetV2":
                    student_model = load_mobilenet_student()
                optimizer = optim.Adam(student_model.parameters(), lr=0.001)
                train_student(teacher_model, student_model, train_loader, test_loader, optimizer, T, alpha, epochs=50, patience=3)
                accuracy = evaluate(student_model, test_loader)
                results[model_name].append((alpha, T, accuracy))
                print(f"Accuracy: {accuracy:.2f}%")
    return results

# 参数范围
alpha_values = [0.2, 0.5, 0.8]
temperature_values = [1.0, 2.0, 4.0]

# 运行测试
results = test_different_parameters(teacher_model, train_loader, test_loader, alpha_values, temperature_values)

# 打印结果
print("\nFinal Results:")
for model_name, model_results in results.items():
    print(f"\nResults for {model_name}:")
    for alpha, T, acc in model_results:
        print(f"Alpha: {alpha}, T: {T}, Accuracy: {acc:.2f}%")
