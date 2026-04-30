import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))

# Spine
ax.plot([0, 9], [0, 0], linewidth=2)

# Problem (Head)
ax.text(9.5, 0, "Low Conceptual\nUnderstanding in DSA",
        va='center', fontsize=11, fontweight='bold')

# ---------------- TOP BONES ----------------
top_positions = [2, 4, 6]

top_data = [
    ("System Design", ["No integration", "No sync", "Separate modules"]),
    ("Technical Limits", ["No mapping", "No states", "Low real-time"]),
    ("Learning Method", ["Theory heavy", "Static examples", "No steps"])
]

for x, (title, subs) in zip(top_positions, top_data):
    # bone
    ax.plot([x, x-0.8], [0, 1.3], linewidth=2)

    # title (above)
    ax.text(x-1.2, 1.6, title, fontsize=10, fontweight='bold')

    # subpoints (shifted LEFT to avoid line)
    for i, sub in enumerate(subs):
        ax.text(x-1.8, 1.2 - i*0.3, f"- {sub}", fontsize=9)

# ---------------- BOTTOM BONES ----------------
bottom_positions = [3, 7]

bottom_data = [
    ("Feedback", ["No hints", "Delayed feedback", "No adaptivity"]),
    ("User Interaction", ["High load", "Low engagement", "Passive learning"])
]

for x, (title, subs) in zip(bottom_positions, bottom_data):
    # bone
    ax.plot([x, x-0.8], [0, -1.3], linewidth=2)

    # title (below)
    ax.text(x-1.2, -1.7, title, fontsize=10, fontweight='bold')

    # subpoints (shifted LEFT to avoid line)
    for i, sub in enumerate(subs):
        ax.text(x-1.8, -1.1 + i*0.3, f"- {sub}", fontsize=9)

# Title
plt.title("Fishbone Diagram – DASHED Root Cause Analysis",
          fontsize=14, fontweight='bold')

# Clean layout
ax.set_xlim(-1, 10)
ax.set_ylim(-2.5, 2.5)
ax.axis('off')

plt.tight_layout()
plt.savefig("dashed_fishbone_compact.png", dpi=300)
plt.show()