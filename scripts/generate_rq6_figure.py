#!/usr/bin/env python3
"""Generate RQ6 overhead scaling figure."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# Data from experiments
budgets = [5, 10, 20, 50]

# CBR overhead (ms)
cbr_small = [0.50, 0.85, 1.34, 2.81]
cbr_medium = [0.29, 0.58, 1.16, 2.84]
cbr_large = [0.36, 0.73, 1.44, 3.59]

# BBL overhead (constant)
bbl_small = 0.09
bbl_medium = 0.13
bbl_large = 0.13

# Combined overhead
combined_small = [c + bbl_small for c in cbr_small]
combined_medium = [c + bbl_medium for c in cbr_medium]
combined_large = [c + bbl_large for c in cbr_large]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: CBR overhead scaling
ax1.plot(budgets, cbr_small, 'o-', linewidth=2, markersize=8,
         label='Small', color='#2E86AB')
ax1.plot(budgets, cbr_medium, 's-', linewidth=2, markersize=8,
         label='Medium', color='#A23B72')
ax1.plot(budgets, cbr_large, '^-', linewidth=2, markersize=8,
         label='Large', color='#F18F01')

ax1.set_xlabel('CBR Sampling Budget', fontsize=12)
ax1.set_ylabel('CBR Overhead (ms)', fontsize=12)
ax1.set_title('(a) CBR Overhead Scales Linearly with Budget', fontsize=13)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(budgets)

# Plot 2: Combined overhead
ax2.plot(budgets, combined_small, 'o-', linewidth=2, markersize=8,
         label='Small', color='#2E86AB')
ax2.plot(budgets, combined_medium, 's-', linewidth=2, markersize=8,
         label='Medium', color='#A23B72')
ax2.plot(budgets, combined_large, '^-', linewidth=2, markersize=8,
         label='Large', color='#F18F01')

ax2.set_xlabel('CBR Sampling Budget', fontsize=12)
ax2.set_ylabel('Total Overhead (CBR + BBL, ms)', fontsize=12)
ax2.set_title('(b) Total Validation Overhead', fontsize=13)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(budgets)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = 'results/rq6/fig_rq6_overhead.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

# Also save as PNG for preview
output_png = 'results/rq6/fig_rq6_overhead.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Preview saved to {output_png}")

plt.close()

# Print statistics
print("\n" + "="*60)
print("Overhead Scaling Analysis")
print("="*60)
print(f"CBR overhead range: {min(cbr_small+cbr_medium+cbr_large):.2f}--{max(cbr_small+cbr_medium+cbr_large):.2f} ms")
print(f"BBL overhead range: {min([bbl_small, bbl_medium, bbl_large]):.2f}--{max([bbl_small, bbl_medium, bbl_large]):.2f} ms")
print(f"Total overhead range: {min(combined_small+combined_medium+combined_large):.2f}--{max(combined_small+combined_medium+combined_large):.2f} ms")
print(f"\nAt default budget (20):")
print(f"  Small:  {combined_small[2]:.2f} ms (CBR: {cbr_small[2]:.2f}, BBL: {bbl_small:.2f})")
print(f"  Medium: {combined_medium[2]:.2f} ms (CBR: {cbr_medium[2]:.2f}, BBL: {bbl_medium:.2f})")
print(f"  Large:  {combined_large[2]:.2f} ms (CBR: {cbr_large[2]:.2f}, BBL: {bbl_large:.2f})")
print("\nScaling factor (budget 5 → 50):")
for name, data in [('Small', cbr_small), ('Medium', cbr_medium), ('Large', cbr_large)]:
    ratio = data[3] / data[0]
    print(f"  {name}: {ratio:.1f}× (expected: 10× for linear)")
