# 📝 Next Word Prediction with LSTM
**Deep Learning for Predictive Language Modeling**  
_LSTM neural network trained on classic literature to predict the next word in a sequence!_

## 🚀 Project Overview
This project demonstrates a complete pipeline for building an intelligent **Next Word Predictor** using deep learning. Leveraging **LSTM networks** and a large corpus (“The Adventures of Sherlock Holmes”), the model learns context and language structure—offering near-instant next-word suggestions! Trained on 100k+ samples and 8200-word vocabulary.

## 🌟 Features
- **Text Preprocessing:** Clean and tokenize raw literature.
- **Sequence Generator:** Generates datasets for 3-word input → 1-word prediction.
- **State-of-the-Art Model:** Two stacked LSTM layers, word embeddings, dense softmax output.
- **Training & Checkpointing:** Trains for up to 70 epochs; best model autosaved.
- **Interactive Prediction:** Enter custom lines & see predicted next word instantly.
- **Fully Reusable:** Scripted for use with **any English text corpus**.

## 🗂️ File Structure
├── nextword.txt            
├── next_words.keras        
├── token.pkl               
├── README.md              
├── (your script files)


## 📊 Model Summary
- Embedding Layer: 8200 words × 10 dimensions
- Two LSTM Layers: 512 units each
- Dense Layer: 512 neurons (ReLU)
- Output: 8200-way softmax (word probabilities)
- **~7.7M trainable parameters**
- Trained up to: **70 epochs**

## ⚡ Quick Start
1. **Clone this Repo**
    ```bash
    git clone https://github.com/yourusername/next-word-lstm.git
    cd next-word-lstm
    ```
2. **Install Requirements**
    ```bash
    pip install tensorflow numpy pickle
    # (If running on Colab, upload your dataset as shown in code.)
    ```
3. **Train the Model (or use pre-trained)**
    - Run the provided Jupyter notebook or Python script.
4. **Use the Predictor**
    ```python
    # Example usage
    Predict_Next_Words(model, tokenizer, ['your', 'input', 'words'])
    ```

## 🎮 Example Interaction
```
Enter your line: To Sherlock Holmes
['To', 'Sherlock', 'Holmes']
esq

Enter your line: love for Irene
['love', 'for', 'Irene']
adler

Enter your line: perfect reasoning and
['perfect', 'reasoning', 'and']
observing
```

## 🧠 How It Works

1. **Prepare Data:**  
   - Clean raw text—strip special characters and excess spaces.
   - Generate 4-word sliding windows: (w1,w2,w3) ➜ (w4).
2. **Tokenize:**  
   - Convert all words to integer tokens and one-hot encode outputs.
3. **Model:**  
   - 2× stacked LSTM layers remember word context.
   - Dense layers produce probabilities for every vocabulary word.
4. **Prediction:**  
   - Enter **any** three-word phrase.
   - Model predicts most probable next word in context.

## 📈 Results
- **Training Loss dropped from ~6.7 to < 0.5 over 70 epochs**
- Realistic, in-context next word completions—mirrors original literature.

## 🧩 Customization
- Swap `nextword.txt` with any English text corpus for personalized training!
- Adjust sequence length or LSTM size for deeper context or memory.

