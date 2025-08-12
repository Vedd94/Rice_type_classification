# PyTorch Rice Type Classification Project

## 📌 Overview
This project implements a **deep learning classifier** using PyTorch to classify rice types based on tabular data. The model is built with Artificial Neural Networks (ANN) following standard PyTorch practices.

## 🎯 Key Features
- **Data Preprocessing**: Handles missing values and normalizes features
- **ANN Architecture**: Customizable hidden layers with sigmoid activation
- **Training Pipeline**: Includes training, validation, and test splits
- **Evaluation Metrics**: Tracks loss and accuracy across epochs
- **GPU Support**: Automatically utilizes CUDA if available

## 🛠️ Technical Details
### Model Architecture
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()
```

### Training Specifications
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Batch Size**: 32
- **Epochs**: 10
- **Train/Val/Test Split**: 70%/15%/15%

## 📂 Dataset
- **Source**: [Kaggle Rice Type Classification](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification)
- **Features**: Multiple agricultural measurements
- **Target**: Binary rice classification

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install torch opendatasets pandas scikit-learn matplotlib
   ```

2. Execute the script:
   ```bash
   python pytorch_fcc_tabular_classification.py
   ```

## 📊 Performance Metrics
The model outputs:
- Training/validation loss and accuracy per epoch
- Final test accuracy (printed after training)

## 💡 Potential Improvements
1. Experiment with deeper architectures
2. Try different activation functions (ReLU, LeakyReLU)
3. Implement early stopping
4. Add learning rate scheduling

## 📜 License
MIT License - Free for academic and commercial use

---
Developed by Ved Dahale  
For educational and research purposes
