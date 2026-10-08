"""
Practical 10: Design a model to detect Fake News from Multimodal data
Multimodal = more than one type of data. Here: TEXT (headline / post) + IMAGE.

Pipeline : text -> TF-IDF features
           image -> colour features (default) or ResNet18 deep features (--deep)
           early fusion = join both feature vectors -> Logistic Regression
Compares : text-only model, image-only model and the fused multimodal model.

Usage
  1) Quick demo with synthetic data (no download needed):
         python fake_news_multimodal.py
  2) Real data (a CSV with columns: text, image, label  where label 0 = real, 1 = fake,
     and 'image' is a file name inside the image folder):
         python fake_news_multimodal.py --csv data.csv --img_dir images/
  3) Add --deep to use ResNet18 image features (needs torch + torchvision, internet once).
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # remove this line if you run in Jupyter / Colab
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Multimodal (text + image) fake news detection")
parser.add_argument("--csv", help="CSV file with columns: text, image, label (0 = real, 1 = fake)")
parser.add_argument("--img_dir", default=".", help="folder containing the images named in the CSV")
parser.add_argument("--deep", action="store_true", help="use ResNet18 deep image features")
args = parser.parse_args()
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
def make_demo_data(n=600):
    """Small SYNTHETIC dataset so the pipeline can run without downloading anything.
    Text and image each carry only a partial, noisy signal about the label."""
    fake_words = ["shocking", "secret", "miracle", "exposed", "unbelievable",
                  "hoax", "conspiracy", "banned", "leaked", "cure"]
    real_words = ["officials", "report", "study", "announced", "according",
                  "council", "university", "budget", "research", "statement"]
    neutral = ["the", "city", "new", "today", "people", "government", "after", "year",
               "local", "market", "health", "school", "police", "water", "train",
               "road", "team", "court"]
    texts, images, labels = [], [], []
    for _ in range(n):
        label = int(RNG.random() < 0.5)                       # 1 = fake, 0 = real
        p = 0.65 if label == 1 else 0.35                      # chance a signal word is a 'fake' word
        words = list(RNG.choice(neutral, 7))
        for _ in range(3):
            words.append(RNG.choice(fake_words) if RNG.random() < p else RNG.choice(real_words))
        RNG.shuffle(words)
        texts.append(" ".join(words))

        base = RNG.normal(120, 40, (32, 32, 3))
        red_tint = RNG.random() < p                           # fake images lean red, real lean blue
        base[:, :, 0 if red_tint else 2] += 30
        base += RNG.normal(0, 15)                             # random brightness change
        images.append(Image.fromarray(np.clip(base, 0, 255).astype("uint8")))
        labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels}), images

if args.csv:
    df = pd.read_csv(args.csv)
    images, keep = [], []
    for i, name in enumerate(df["image"]):
        path = os.path.join(args.img_dir, str(name))
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img.thumbnail((224, 224))
            images.append(img)
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    DATA_NOTE = f"Real data loaded from {args.csv}"
else:
    df, images = make_demo_data()
    DATA_NOTE = "DEMO MODE: synthetic data, results only show how the pipeline works"

print(DATA_NOTE)
print("Samples:", len(df), "| Fake:", int((df.label == 1).sum()), "| Real:", int((df.label == 0).sum()))
print("Example text:", df.text.iloc[0])
print("Missing text values:", int(df.text.isnull().sum()))
df["text"] = df["text"].fillna("")

# ---------------------------------------------------------------
# Step 2: Clean the text
# ---------------------------------------------------------------
def clean(t):
    t = re.sub(r"http\S+", " ", str(t).lower())     # lower-case, remove links
    return re.sub(r"[^a-z\s]", " ", t)              # keep letters only

df["clean_text"] = df["text"].apply(clean)

# ---------------------------------------------------------------
# Step 3: Split into training and testing sets (same split for all models)
# ---------------------------------------------------------------
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=df["label"])
y = df["label"].values
y_train, y_test = y[tr_idx], y[te_idx]
print("Training samples:", len(tr_idx), "| Testing samples:", len(te_idx))

# ---------------------------------------------------------------
# Step 4: Text features (TF-IDF)
# ---------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
T_train = tfidf.fit_transform(df["clean_text"].iloc[tr_idx])
T_test = tfidf.transform(df["clean_text"].iloc[te_idx])
print("Text feature size:", T_train.shape[1])

# ---------------------------------------------------------------
# Step 5: Image features
# ---------------------------------------------------------------
def colour_features(img):
    """Mean, standard deviation and an 8-bin histogram for each RGB channel (30 numbers)."""
    a = np.asarray(img.resize((64, 64)), dtype=float)
    feats = []
    for c in range(3):
        ch = a[:, :, c]
        hist, _ = np.histogram(ch, bins=8, range=(0, 255))
        feats += [ch.mean(), ch.std()] + list(hist / ch.size)
    return feats

def deep_features(imgs):
    """512 numbers per image from a ResNet18 pretrained on ImageNet."""
    import torch
    from torchvision import models
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval()
    prep = weights.transforms()
    out = []
    with torch.no_grad():
        for im in imgs:
            out.append(net(prep(im).unsqueeze(0)).squeeze(0).numpy())
    return np.array(out)

I_all = deep_features(images) if args.deep else np.array([colour_features(im) for im in images])
print("Image feature type:", "ResNet18 deep features" if args.deep else "colour statistics",
      "| size:", I_all.shape[1])

scaler = StandardScaler().fit(I_all[tr_idx])
I_train, I_test = scaler.transform(I_all[tr_idx]), scaler.transform(I_all[te_idx])

# ---------------------------------------------------------------
# Step 6: Train three models: text only, image only, fused (text + image)
# ---------------------------------------------------------------
def fit_eval(name, Xtr, Xte):
    model = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_train)
    pred = model.predict(Xte)
    return model, pred, accuracy_score(y_test, pred), f1_score(y_test, pred)

text_model, p_text, acc_text, f1_text = fit_eval("text", T_train, T_test)
img_model, p_img, acc_img, f1_img = fit_eval("image", I_train, I_test)

F_train = hstack([T_train, csr_matrix(I_train)]).tocsr()      # early fusion: join the features
F_test = hstack([T_test, csr_matrix(I_test)]).tocsr()
fused_model, p_fused, acc_fused, f1_fused = fit_eval("fused", F_train, F_test)

results = pd.DataFrame({
    "Model": ["Text only", "Image only", "Fused (text + image)"],
    "Accuracy": [acc_text, acc_img, acc_fused],
    "F1 (fake)": [f1_text, f1_img, f1_fused],
}).round(4)
print("\nModel comparison on the test set:")
print(results.to_string(index=False))

# ---------------------------------------------------------------
# Step 7: Detailed evaluation of the fused model
# ---------------------------------------------------------------
print("\nClassification report (fused model):")
print(classification_report(y_test, p_fused, target_names=["real", "fake"]))
cm = confusion_matrix(y_test, p_fused)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

words = np.array(tfidf.get_feature_names_out())
order = np.argsort(text_model.coef_[0])
print("\nWords most linked to REAL news :", ", ".join(words[order[:8]]))
print("Words most linked to FAKE news :", ", ".join(words[order[-8:][::-1]]))

# ---------------------------------------------------------------
# Step 8: Plots
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(3); w = 0.38
ax.bar(x - w/2, results["Accuracy"], w, label="Accuracy")
ax.bar(x + w/2, results["F1 (fake)"], w, label="F1 (fake)")
ax.set_xticks(x); ax.set_xticklabels(results["Model"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Text-only vs Image-only vs Multimodal (fused)")
for i, (a, f) in enumerate(zip(results["Accuracy"], results["F1 (fake)"])):
    ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

plt.figure(figsize=(5.2, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["real", "fake"], yticklabels=["real", "fake"])
plt.title("Confusion Matrix - Multimodal model")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------
print(f"\n{DATA_NOTE}")
print(f"Conclusion: text-only accuracy = {acc_text:.2f}, image-only = {acc_img:.2f}, "
      f"fused multimodal = {acc_fused:.2f}.")
