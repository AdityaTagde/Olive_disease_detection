## 🪴 Olive Leaf Disease Detection using Deep Learning

### 🌱 Overview

This project is an AI-powered web application that detects **olive leaf diseases** from uploaded images.
It uses a **Convolutional Neural Network (CNN)** trained on olive leaf datasets and a **Streamlit** front-end for real-time inference.

The goal is to help farmers and agricultural researchers quickly identify and manage plant diseases for healthier olive production.

---

### 🚀 Features

* 📸 Upload an olive leaf image (JPG, JPEG, or PNG)
* 🤖 Detects whether the leaf is **Healthy**, **Aculus olearius**, or **Olive Peacock Spot**
* 💡 Real-time predictions powered by TensorFlow
* 🎨 Beautiful dark Streamlit interface with smooth UI
* 📊 Confidence score for each prediction

---

### 🧠 Model Details

* Framework: **TensorFlow / Keras**
* Type: **Convolutional Neural Network (CNN)**
* Input size: **224 × 224 × 3**
* Output classes: `['Healthy', 'Aculus olearius', 'Olive Peacock Spot']`
* Model file: `olive_model.h5`

---

### 🛠️ Tech Stack

| Component      | Technology        |
| -------------- | ----------------- |
| Frontend       | Streamlit         |
| Backend        | Python            |
| Deep Learning  | TensorFlow, Keras |
| Image Handling | Pillow, NumPy     |

---

### ⚙️ Installation

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/AdityaTagde/olive-leaf-disease-detection.git
cd olive-leaf-disease-detection
```

#### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For macOS/Linux
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run the app

```bash
streamlit run olive_app.py
```

---

### 📂 Project Structure

```
olive-leaf-disease-detection/
│
├── olive_app.py              # Streamlit app
├── olive_model.h5            # Trained CNN model
├── Olive_disease_detection.ipynb  # Jupyter notebook (model training)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

### 💻 Example Output

* Upload image → model analyzes → returns:

  * 🌿 **Healthy**
  * 🍂 **Aculus olearius**
  * 🍃 **Olive Peacock Spot**

You’ll also see a confidence percentage and a sleek prediction box in a dark-themed UI.

---
### 🖼️ Screenshots

#### 🧩 Application Interface
![App Screenshot](https://github.com/AdityaTagde/Olive_disease_detection/o1.png)

#### 🌿 Prediction Example
![Prediction Screenshot](https://github.com/AdityaTagde/Olive_disease_detection/o2.png)

![Prediction Screenshot](https://github.com/AdityaTagde/Olive_disease_detection/o3.png)

---

### 🧩 Requirements

Make sure you have:

* Python 3.10+
* TensorFlow 2.15.0
* Streamlit 1.27+
* NumPy, Pillow

---
### 🏁 Future Improvements

* Add more olive leaf disease classes
* Integrate Grad-CAM visualization
* Build mobile-friendly UI
* Deploy with TensorFlow Lite for on-device predictions

---

### 🩶 Acknowledgments

* Olive leaf disease dataset (source: Kaggle / research dataset)
* TensorFlow and Streamlit teams for their amazing open-source tools

---

