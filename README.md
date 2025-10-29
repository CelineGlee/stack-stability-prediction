# Block Stack Stability Prediction  

Predicting the stability of block stacks from single images using computer vision and deep learning.  

## Project Overview  
This project addresses a **visual physical reasoning** task: predicting the **stable height** of block stacks, where the stable portion remains balanced.  
We use the **ShapeStacks dataset** and apply transfer learning with pre-trained computer vision models (EfficientNet, ResNet).  

My solution combines **data preprocessing, augmentation, and error analysis** to improve performance. The final model achieved **~61% accuracy** on Kaggle submissions, compared to the baseline of 40%.  

## Example Data  
*Examples from the ShapeStacks dataset. Each image shows stacks of 2–6 blocks of various shapes (cubes, cylinders, spheres, rectangular solids) under different lighting and camera angles. The dataset also provides metadata such as instability type, camera angle, and stable height.* 

![10421](https://github.com/user-attachments/assets/35917e4f-c579-406e-a339-7e6bf3b7bd89) 
![16444](https://github.com/user-attachments/assets/862824d2-9544-4aa0-a11b-764fea3fb01d)
![33436](https://github.com/user-attachments/assets/89890a10-b016-4f7f-903e-019e7d3cfebf)

## Methodology  

1. **Data Preprocessing**  
   - Resize images to 224×224  
   - Normalize pixel values  
   - Standardize with ImageNet mean/std  

2. **Data Augmentation**  
   - Brightness adjustment  
   - Shifts, shear, zoom  
   - Horizontal flips  

3. **Modeling**  
   - Transfer learning with EfficientNet, ResNet, VGG  
   - Fully connected layers with dropout  
   - Softmax output for stable height classification  

4. **Training**  
   - 80/20 split for train/validation  
   - Adam optimizer, learning rate = 0.001  
   - Early stopping + checkpointing  

## Results  

- Baseline (no augmentation): **56% validation accuracy**  
- With augmentation: **59% validation accuracy**  
- Final EfficientNet model: **61% test accuracy (Kaggle submission)**

| Model             | Accuracy | Notes                                |
|-------------------|----------|--------------------------------------|
| Baseline (no aug) | 56%      | Simple preprocessing only             |
| + Augmentation    | 59%      | Improved robustness and lower loss    |
| EfficientNetB0    | **61%**  | Final Kaggle submission, best result  |
 

**Error analysis:**  
- Model tends to **underestimate stable height**  
- Performs better on lower stable heights  
- Grad-CAM showed focus on block bases and shapes  

## Error Analysis  
- **Bias toward lower stable heights** → underestimation errors  
- **Class imbalance** → fewer examples for taller stacks  
- **Background noise** affects predictions (segmentation didn’t improve results)  

