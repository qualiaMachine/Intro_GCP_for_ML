#!/usr/bin/env python3
"""data_check.py — verify that HTCondor can transfer and read a CSV file."""

import pandas as pd
import os

print(f"Working directory: {os.getcwd()}")
print(f"Files available: {os.listdir('.')}")

df = pd.read_csv("titanic_train.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSurvival rate: {df['Survived'].mean():.2%}")

# Write a small output file to confirm output transfer works
with open("data_summary.txt", "w") as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(f"Survival rate: {df['Survived'].mean():.2%}\n")

print("\nWrote data_summary.txt")
