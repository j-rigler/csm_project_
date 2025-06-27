
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pathlib import Path


production_cap = False
scenario = ["ALL", "HOA", "RUS", "URU", "PAK"]

current_dir = Path.cwd()
print(f"Current working directory:", current_dir)

loss_calculation_folder = current_dir / "evaluation"
print("input files at",loss_calculation_folder)

output_dir = os.path.join(
    loss_calculation_folder,
    "mit_cap" if production_cap else "ohne_cap")
print("output directory at", output_dir)


df = {}
for s in scenario:

    if production_cap:
        filename = loss_calculation_folder / f"{s}-Losses_capped.csv"  
    
    else:
        filename = loss_calculation_folder / f"{s}-Losses_no_cap.csv"

    if filename.exists():
        try:
            df[s] = pd.read_csv(filename)
            print(f"✓ Loaded: {filename}")
        except Exception as e:
            print(f"× Error loading {filename}: {str(e)}")
    else:
        print(f"! File not found: {filename}")

sum_RUS = df['RUS']['absolute_losses [t]'].sum()
sum_PAK = df['PAK']['absolute_losses [t]'].sum()
sum_URU = df['URU']['absolute_losses [t]'].sum()
sum_HOE = df['HOA']['absolute_losses [t]'].sum()

sum_all = sum_RUS + sum_PAK + sum_URU + sum_HOE
print(f"Total absolute losses for RUS, PAK, URU, HOA: {sum_all:.0f} t")
sum_ALL = df['ALL']['absolute_losses [t]'].sum()
print(f"Total absolute losses for ALL: {sum_ALL:.0f} t")
difference = sum_ALL - sum_all
print(difference)

#plot
sn.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

bars = [
    {"label": "RUS + PAK + URU + HOA", "value": sum_all},
    {"label": "Compound shock (ALL)", "value": sum_ALL}
]

labels = [b["label"] for b in bars]
values = [b["value"] for b in bars]

sn.barplot(x=labels, y=values, palette="muted", ax=ax)
ax.set_ylabel("Total Absolute Losses [t]")
ax.set_title("Global Food Losses: Individual vs Compound Shocks")
for i, v in enumerate(values):
    ax.text(i, v + 0.01 * max(values), f"{v:,.0f} t", ha='center', va='bottom')

plt.tight_layout()
plt.show()