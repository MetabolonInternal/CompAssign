#!/usr/bin/env python
"""
Test script to verify the decoy compound bug in evaluation metrics.
This script will demonstrate that decoy assignments are not being counted as false positives.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scripts.create_synthetic_data import create_metabolomics_data

def check_decoy_assignments():
    """Check if the model is assigning peaks to decoy compounds and if metrics reflect this."""
    
    # Check if we have results from a recent run
    results_dir = Path("output/results")
    if not results_dir.exists():
        print("❌ No results directory found. Please run training first.")
        return
    
    assignments_file = results_dir / "peak_assignments.csv"
    if not assignments_file.exists():
        print("❌ No peak assignments file found. Please run training first.")
        return
    
    # Load assignments
    assignments = pd.read_csv(assignments_file)
    
    # Load saved compound info instead of regenerating
    compound_info_file = results_dir / "compound_info.csv"
    if not compound_info_file.exists():
        print("❌ No compound info file found. Please run training with the updated code.")
        return
    
    compound_df = pd.read_csv(compound_info_file)
    
    # Get decoy and real compound IDs
    decoy_ids = set(compound_df[compound_df['is_decoy']]['compound_id'].values)
    real_ids = set(compound_df[~compound_df['is_decoy']]['compound_id'].values)
    
    # Check assignments
    assigned_compounds = assignments['assigned_compound'].dropna()
    if len(assigned_compounds) == 0:
        print("⚠️ No compounds were assigned to any peaks.")
        return
    
    assigned_ids = set(assigned_compounds.unique().astype(int))
    
    # Calculate statistics
    real_assigned = assigned_ids & real_ids
    decoys_assigned = assigned_ids & decoy_ids
    
    # Count how many peaks were assigned to decoys vs real compounds
    peaks_to_real = 0
    peaks_to_decoys = 0
    for _, row in assignments.iterrows():
        if pd.notna(row['assigned_compound']):
            compound_id = int(row['assigned_compound'])
            if compound_id in real_ids:
                peaks_to_real += 1
            elif compound_id in decoy_ids:
                peaks_to_decoys += 1
    
    print("=" * 60)
    print("DECOY ASSIGNMENT ANALYSIS")
    print("=" * 60)
    print(f"\nCompound Library:")
    print(f"  Total compounds: {len(compound_df)}")
    print(f"  Real compounds: {len(real_ids)} {list(sorted(real_ids))[:5]}...")
    print(f"  Decoy compounds: {len(decoy_ids)} {list(sorted(decoy_ids))[:5]}...")
    
    print(f"\nAssignment Results:")
    print(f"  Unique compounds assigned: {len(assigned_ids)}")
    print(f"  Real compounds assigned: {len(real_assigned)} {list(sorted(real_assigned))[:5]}...")
    print(f"  🐛 DECOY compounds assigned: {len(decoys_assigned)} {list(sorted(decoys_assigned))[:5]}...")
    
    print(f"\nPeak Assignments:")
    print(f"  Total peaks assigned: {len(assigned_compounds)}")
    print(f"  Peaks → real compounds: {peaks_to_real}")
    print(f"  Peaks → decoy compounds: {peaks_to_decoys} ⚠️")
    
    # Calculate what metrics SHOULD be
    true_positives = len(real_assigned)
    false_positives = len(decoys_assigned)
    total_predicted = true_positives + false_positives
    
    if total_predicted > 0:
        true_precision = true_positives / total_predicted
        print(f"\n📊 Metrics Impact:")
        print(f"  True Positives (real compounds): {true_positives}")
        print(f"  False Positives (decoys): {false_positives}")
        print(f"  ACTUAL Precision: {true_precision:.1%}")
        
        # Load reported metrics
        import json
        metrics_file = results_dir / "assignment_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            reported_precision = metrics.get('compound_precision', 0)
            print(f"  REPORTED Precision: {reported_precision:.1%}")
            
            if abs(reported_precision - true_precision) > 0.05:
                print(f"\n❌ BUG DETECTED: Reported precision ({reported_precision:.1%}) doesn't match actual ({true_precision:.1%})")
                if len(decoys_assigned) > 0:
                    print(f"   The model assigned {peaks_to_decoys} peaks to {len(decoys_assigned)} decoy compounds")
                    print(f"   These false positives are NOT being counted in the metrics!")
            else:
                print(f"\n✅ METRICS CORRECT: Reported precision matches actual precision!")
                if len(decoys_assigned) == 0:
                    print(f"   Model correctly avoided assigning peaks to decoy compounds")
                else:
                    print(f"   Model correctly counted {len(decoys_assigned)} decoy assignments as false positives")
    else:
        print("\n⚠️ No compounds were assigned, cannot calculate precision.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_decoy_assignments()