import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

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
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout rate
        self.fc2 = nn.Linear(1024, n_labels)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu5(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_model(path, num_classes=5):
    # Add map_location=torch.device('cpu') to load the model on CPU
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    # Instantiate the ResNet-18 model with custom number of output classes
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(state_dict)
    
    return model

#preprocess each image in the same format as done in the model
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)

    # Check if the image has an alpha channel (RGBA)
    if input_tensor.shape[0] == 4:
        # Remove the alpha channel
        input_tensor = input_tensor[:3, :, :]

    # Normalize the image
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    input_tensor = normalize(input_tensor)

    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# Function to make predictions
def predict_image_class(image, model):
    # Preprocess the image
    input_batch = preprocess_image(image)
    # Perform inference
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main():
    #load main model
    model_save_path = 'best_model.pth'
    try:
        state_dict = torch.load(model_save_path, map_location=torch.device('cpu'))
        print("Model state dictionary loaded successfully.")
    
        # load the state dictionary into a new model instance
        model = NeuralNet(15).to(device)
        model.float()
        model.load_state_dict(state_dict)
        print("Model loaded successfully with the state dictionary.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # Page Title
    st.set_page_config(layout="wide")


    
    # st.title('Welcome to DinoClassify - Journey Through the Mesozoic Era')
    # #explain project
    # st.text('Embark on a prehistoric adventure with DinoClassify, our state-of-the-art dinosaur image classification project. Leveraging the powers of modern machine learning, we\'ve brought ancient creatures to life through the lens of technology.')
    # st.subheader('Project Highlights:')
    # st.text('Extensive Dataset: Our journey started with a modest collection of 200 dinosaur images. Recognizing the need for more data, we meticulously expanded our repository to 840 images across five distinct dinosaur classes.')
    # st.text('Advanced Augmentation: Due to the rarity of dinosaur images, we turned to creative augmentation techniques, enhancing our dataset to 6469 images with transformations that mimic nature\'s diversity—shifts, zooms, flips, and more.')
    # st.text('Innovative Model Training: With a curated dataset in place, we harnessed the capabilities of a ResNet-18 neural network, meticulously fine-tuning it to become a connoisseur of Cretaceous creatures.')
    # st.text('Optimized Performance: After rigorous training and validation, we\'ve honed our model to an impressive 90 percent accuracy, uncovering patterns unseen since the dinosaurs roamed the earth.')
    # #upload and explain stats
    # st.text('Our Confusion Matrix reveals the intricacies of our model\'s predictions, illuminating the path it took to distinguish an Ankylosaurus from a Tyrannosaurus.')
    # st.image('Confusion_Matrix.png')
    # st.text('The Precision and Recall Scores are a testament to our model\'s discerning eye, quantifying its ability to classify with both accuracy and consistency.')
    # st.image('Classification_Report_Image.png')
    # st.text('The graph below captures the ebb and flow of our training journey, with a special highlight on the epoch that set a new benchmark in dinosaur recognition.')
    # st.image('Epoch_Graph.png')
    
    # st.text('Upload an image below to learn more abouta dinosaur:')
    # # File uploader
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # st.markdown("[References](https://avarshney15.github.io/datahacks/References)")
    # # Prediction
    # if uploaded_file is not None:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption='Uploaded Image', use_column_width=True)
    #     if st.button('Predict'):
    #         image_class = predict_image_class(image, model_clone)
    #         dino_names = ['Ankylosaurus', 'Brontosaurus', 'Pterodactyl', 'Trex', 'Triceratops']
            
            
    #         external_url = f"https://avarshney15.github.io/datahacks/{dino_names[image_class]}"
            
    #         st.markdown(f'<meta http-equiv="refresh" content="0;URL={external_url}">', unsafe_allow_html=True)
            
            
    #         st.markdown("[References](https://avarshney15.github.io/datahacks/References)")
    #         #st.switch_page(f"pages/{dino_names[image_class]}.py")

    #         #st.write('Predicted Class:', dino_names[image_class])
            

if __name__ == '__main__':
    main()