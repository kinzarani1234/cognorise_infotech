# CognoRise Infotech AI Internship Projects
This repository contains the projects developed during my AI internship at CognoRise Infotech. The main project focuses on Chest X-ray COVID-19 & Pneumonia Detection.

# Project Description
This project uses convolutional neural networks (CNNs) to detect COVID-19 and Pneumonia from chest X-ray images. The trained model can classify images into three categories: COVID Positive, Pneumonia Positive, and Normal.

# Setup Instructions
### Step 1: Clone or Download the Repository
You can download the ZIP folder of this repository and extract it.
git clone https://github.com/your-repository-link.git

### Step 2: Download the Dataset
Download the dataset from Kaggle:
data set: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
Chest X-ray COVID-19 Pneumonia Dataset
After downloading the dataset, extract it, and make sure the dataset folder name is Data.

### Step 3: Organize the Dataset
Once you have extracted the dataset and  place this folder in the same directory where the repository’s code files are located. This will ensure the app and model can access the dataset.

For example:

project-folder/

    ├── app.py
    
    ├── best_model1.keras
    
    ├── covid19.ipynb
    
    ├── requirements.txt
    
    └── Data/  # <-- Place the extracted dataset here
### Step 4: Set Up the Environment
Create a virtual environment and install the necessary libraries.

##### Create a virtual environment (optional but recommended)
python -m venv venv

##### Activate the virtual environment
###### On Windows:
venv\Scripts\activate
###### On MacOS/Linux:
source venv/bin/activate

##### Install dependencies
pip install -r requirements.txt

### Step 5: Train and Save the Model
Open the provided Jupyter notebook (covid19.ipynb) and run all the cells.
After training, the model will be saved as best_model1.keras.

### Step 6: Run the Streamlit App
To deploy the app locally:

Navigate to the folder where app.py is located.
Run the following command in your terminal or command prompt:

###### streamlit run app.py

The app will be served on localhost and can be accessed via a browser.
Libraries Used
Make sure the following libraries are installed (included in requirements.txt):

TensorFlow
Keras
Numpy
Pillow
Streamlit
You can add other dependencies as needed.
for example pandas , matplotlib , seaborn, scikit learn

![image](https://github.com/user-attachments/assets/76204b03-0944-436a-9d34-f276e1fc9130)
![image](https://github.com/user-attachments/assets/a543214c-5ec7-460f-95e6-40ae68a915ee)
![image](https://github.com/user-attachments/assets/c79f2ddd-a80a-45d3-98d1-dec0a4fbd3e3)



