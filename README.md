# Trustworthy ML Robustness Pipeline

A single-file PyTorch pipeline for training a robust 9-class image classifier using ResNet-34, adversarial training, ensemble distillation, model soups, and final robust validation.
The final output is a plain `torchvision` ResNet-34 state dict ready for submission.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place `train.npz` in the same folder as `main.py`.

Expected keys:

```text
images
labels
```

Expected image shape:

```text
(N, 3, 32, 32)
```

## Run

```bash
python main.py --data train.npz --out output
```

The script trains from scratch if no champion checkpoint is provided.

## Output

After training, the final selected model is saved as:

```text
output/output_submission.pt
```

It is also copied to:

```text
output_submission.pt
```

## Submission

Use:

```python
MODEL_PATH = "output_submission.pt"
MODEL_NAME = "resnet34"
```
