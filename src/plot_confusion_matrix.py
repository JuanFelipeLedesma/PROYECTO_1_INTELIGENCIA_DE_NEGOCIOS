import pandas as pd, matplotlib.pyplot as plt
cm = pd.read_csv("evaluation/cv_confusion_matrix.csv", index_col=0)
plt.figure()
plt.imshow(cm.values, interpolation="nearest")
plt.xticks(range(cm.shape[1]), cm.columns, rotation=45, ha="right")
plt.yticks(range(cm.shape[0]), cm.index)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm.values[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("evaluation/cv_confusion_matrix.png", dpi=200)
print("Guardado: evaluation/cv_confusion_matrix.png")
