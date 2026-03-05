"""
compare_all_models.py — Final Combined Leaderboard
==================================================
Merges TCA and SER leaderboards and sorts by F1-macro.
"""

import pandas as pd
from pathlib import Path

def main():
    outputs_dir = Path("./outputs")
    tca_path = outputs_dir / "tca_leaderboard.csv"
    ser_path = outputs_dir / "ser_leaderboard.csv"

    frames = []

    if tca_path.exists():
        df_tca = pd.read_csv(tca_path)
        frames.append(df_tca)
    else:
        print("⚠️ Warning: tca_leaderboard.csv not found.")

    if ser_path.exists():
        df_ser = pd.read_csv(ser_path)
        frames.append(df_ser)
    else:
        print("⚠️ Warning: ser_leaderboard.csv not found.")

    if not frames:
        print("❌ Error: No leaderboard files found to merge.")
        return

    # Combine
    combined = pd.concat(frames, ignore_index=True)

    # Sort by Track and then F1 Macro descending
    combined = combined.sort_values(by=["track", "f1_macro"], ascending=[True, False])

    # Save final
    final_path = outputs_dir / "final_combined_leaderboard.csv"
    combined.to_csv(final_path, index=False)

    print("\n" + "="*90)
    print(f"{'AUDIOGUARD FINAL COMBINED LEADERBOARD':^90}")
    print("="*90)
    print(combined[["track", "model_id", "model_name", "accuracy", "f1_macro"]].to_string(index=False))
    print("="*90)
    
    # Select best models
    print("\n🏆 Recommended Models for Fusion Layer:")
    for track in ["TCA", "SER"]:
        best = combined[combined["track"] == track].iloc[0]
        print(f"  - {track}: {best['model_id']} ({best['model_name']}) with F1 Macro: {best['f1_macro']}")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()
