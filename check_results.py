import json, pandas as pd

print("=" * 60)
print("EXPERIMENT SUMMARY TABLE")
print("=" * 60)
df = pd.read_csv("artifacts/results/summary_results.csv")
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("RAW METRIC CHECKPOINTS")
print("=" * 60)
with open("artifacts/results/raw_results.json") as f:
    d = json.load(f)

print("E1 Clean Validation FPRs:")
print([round(x["fpr"], 4) for x in d["e1"]])

print("\nE3 Losses:")
e3 = d["e3"][0]
print(f"  Clean:      {e3['clean_loss']:.6f}")
print(f"  Undefended: {e3['undefended_loss']:.6f}")
print(f"  Defended:   {e3['defended_loss']:.6f}")

print("\nE2 Sample Detection (Client 0):")
for entry in d["e2"]:
    if entry["client"] == 0:
        print(f"  {entry['attack']:<25} | F1: {entry['f1']:.4f} | FPR: {entry['fpr']:.4f} | Recall: {entry['recall']:.4f}")
