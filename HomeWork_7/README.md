# Homework 7 - Fully Connected Neural Networks for CIFAR-10

This homework studies how depth, width, and activation functions affect a fully connected neural network on CIFAR-10. Six FCNN models were trained and compared using the same data split and evaluation process.

## Dataset

- CIFAR-10: 60,000 RGB images in 10 classes
- Training set: 45,000 images
- Validation set: 5,000 images
- Test set: 10,000 images
- Input size after flattening: 3,072 values
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: cross-entropy loss
- Maximum epochs: 100 with early stopping

The images were standardized with CIFAR-10 channel mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`.

## Experiments

1. Baseline FCNN with two hidden ReLU layers: `3072 -> 512 -> 256 -> 10`
2. Deeper FCNN with four hidden ReLU layers: `3072 -> 512 -> 256 -> 128 -> 64 -> 10`
3. FCNN with ten hidden ReLU layers
4. Wider FCNN: `3072 -> 1024 -> 512 -> 10`
5. Baseline-width FCNN with Sigmoid activation
6. Baseline-width FCNN with Tanh activation

## Final Results

| Model | Best Epoch | Training Time | Training Accuracy | Evaluation Accuracy | Parameters |
|---|---:|---:|---:|---:|---:|
| Baseline ReLU | 3 | 148.49 s | 57.40% | 51.06% | 1,707,274 |
| 4-Hidden-Layer ReLU | 5 | 183.95 s | 63.10% | **53.03%** | 1,746,506 |
| 10-Hidden-Layer ReLU | 8 | 214.90 s | 62.91% | 51.81% | 1,771,466 |
| Wide ReLU | 3 | 144.11 s | 56.66% | 49.80% | 3,676,682 |
| Sigmoid | 7 | 183.83 s | 58.49% | 49.44% | 1,707,274 |
| Tanh | 12 | 244.64 s | 60.07% | 47.03% | 1,707,274 |

## Conclusion

The four-hidden-layer ReLU network performed best with 53.03% evaluation accuracy. Adding more depth improved the baseline at first, but the ten-hidden-layer model was slower and less accurate. Doubling the network width more than doubled the parameter count without improving accuracy. ReLU also performed better than Sigmoid and Tanh.

The training curves showed overfitting in every experiment. Training accuracy continued to increase while validation loss became worse. Early stopping was used to restore the best validation checkpoint. A CNN would likely perform better because it preserves spatial information that is lost when the image is flattened.

## Files

- `Homework_7_Commented.ipynb` - complete notebook with saved outputs and simple comments
- `Homework_7_Report.docx` - editable written report
- `Homework_7_Report.pdf` - final report
- `Homework_7_Code.pdf` - printable code and saved outputs

## Run the Notebook

Open the notebook in Google Colab or Jupyter, install PyTorch and Torchvision if needed, and run the cells from top to bottom. CIFAR-10 downloads automatically through Torchvision.
