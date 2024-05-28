import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, n_labels):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)  # Adjust based on input size
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_labels)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu6(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(path, num_classes):
    model = NeuralNet(num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize to expected input size
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

def predict_image_class(image, model, label_dict):
    input_batch = preprocess_image(image)
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    class_name = label_dict[predicted.item()]
    return class_name

def main():
    st.title('Plant Disease Classification')
    st.text('Upload an image to classify the plant disease.')

    # Dictionary mapping label numbers to class names
    label_dict = {
        0: "Apple scab",
        1: "Black rot",
        2: "Cedar apple rust",
        3: "Healthy",
        4: "Powdery mildew",
        5: "Spot",
        6: "Corn (Maize) Common Rust",
        7: "Blight",
        8: "Grape Esca (Black Measles)",
        9: "Orange Haunglongbing (Citrus greening)",
        10: "Strawberry Leaf scorch",
        11: "Tomato Leaf Mold",
        12: "Tomato Spider mites Two-spotted spider mite",
        13: "Tomato Yellow Leaf Curl Virus",
        14: "Tomato mosaic virus"
    }

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Load model
        model = load_model('best_model-1.pth', 15)
        
        if st.button('Predict'):
            class_name = predict_image_class(image, model, label_dict)
            st.write(f'Predicted class: {class_name}')

if __name__ == '__main__':
    main()
