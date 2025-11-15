# src/utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pairplot(df, hue=None):
    sns.pairplot(df, hue=hue)
    plt.tight_layout()
    plt.show()
