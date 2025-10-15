Overview

The rapid spread of fake news across social media platforms poses a significant threat to public trust, political stability, and public health. While text-based detection methods have made progress, they often fail to capture visual deception — where fake news includes manipulated or misleading images to enhance credibility.

This project presents a deep learning-based approach to identify fake news from image data using Convolutional Neural Network (CNN) architectures such as ResNet50, ResNet18, and VGG16. The goal is to extract discriminative visual features that distinguish authentic images from falsified ones, thereby mitigating the spread of misinformation.

Problem Statement:
Fake News Detection Using Images with Deep Learning
The rapid spread of fake news on social media platforms poses as a serious threat to public trust political stability and public health. Text based Detection methods alone are insufficient as fake news articles increasingly include misleading and manipulated images to enhance credibility. Identify fake news from image data is therefore critical. The challenge is to develop a deep learning based system capable of detecting fake news by analyzing images using convolution neural network architectures such as ResNet50 and ResNet18 and VGG16. By training or fine tuning these models the system seeks to extract meaningful visual features that can help distinguish fake news from real news and curb the negative impact of misinformation.

Traditional fake news detection techniques primarily focus on textual cues, overlooking the visual dimension of misinformation. Fake news creators frequently alter or reuse images to mislead audiences.
Our challenge: Build a CNN-based system that can detect fake news by analyzing images, enabling automated, scalable, and accurate misinformation filtering.

| Category                | Tools / Frameworks                                          |
| ----------------------- | ----------------------------------------------------------- |
| Language                | Python 🐍                                                  |
| Deep Learning Framework | TensorFlow / Keras                                          |
| Models                  | ResNet18, ResNet50, VGG16                                   |
| Libraries               | NumPy, Pandas, Matplotlib, Scikit-learn, OpenCV             |
| Environment             | Jupyter Notebook / Google Colab / Kaggle                    |
| Dataset                 | Fake vs Real News Images Dataset (custom or Kaggle sourced) |

Dataset → Preprocessing → Model Training → Feature Extraction → Classification → Evaluation

Setup Instructions
1. Clone the repository
    git clone https://github.com/yourusername/Fake-News-Detection-Images.git
    cd Fake-News-Detection-Images
2. Install dependencies
   pip install -r requirements.txt
3. Run the Notebook
   Open MINI_PROJECT_TEAM25.ipynb in Jupyter Notebook or Google Colab and execute the cells step-by-step.
4. Run custom training
   You can train your model on a different dataset by updating the dataset path in the notebook.
