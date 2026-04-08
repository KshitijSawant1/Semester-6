import matplotlib.pyplot as plt

# ================= DASHED SDLC TASKS =================
tasks = [
    ("Requirement Gathering", 0, 3),
    ("System Analysis (DFD, ERD)", 3, 6),
    ("System Design", 6, 10),
    ("Frontend Development (React)", 10, 18),
    ("Backend Integration (API/Firebase)", 18, 25),
    ("Testing (Unit + Integration)", 25, 30),
    ("Deployment", 30, 33),
]

# Colors for each phase
colors = [
    "#4CAF50",  # Requirement
    "#2196F3",  # Analysis
    "#9C27B0",  # Design
    "#FF9800",  # Frontend
    "#F44336",  # Backend
    "#00BCD4",  # Testing
    "#000000",  # Deployment
]

# ================= CREATE FIGURE =================
fig, ax = plt.subplots(figsize=(12, 6))

# ================= DRAW BARS =================
for i, (task, start, end) in enumerate(tasks):
    duration = end - start
    ax.barh(i, duration, left=start, color=colors[i], edgecolor="black")

    # Add duration label inside bar
    ax.text(start + duration / 2, i,
            f"{duration}d",
            va='center', ha='center',
            color='white', fontsize=9, fontweight='bold')

# ================= LABELS =================
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task[0] for task in tasks], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Project Timeline (Days)", fontsize=11)
ax.set_title("DASHED Project Gantt Chart (SDLC Phases)", fontsize=14, fontweight='bold')

# ================= REMOVE GRID =================
ax.grid(False)

# ================= STYLE CLEANUP =================
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ================= ADD MILESTONE =================
ax.scatter(33, len(tasks)-1, color="red", s=100, label="Project Complete")
ax.text(33, len(tasks)-1.3, "Milestone", color="red", fontsize=9)

# ================= DEPENDENCY ARROWS =================
for i in range(len(tasks)-1):
    prev_end = tasks[i][2]
    next_start = tasks[i+1][1]

    ax.annotate("",
        xy=(next_start, i+1),
        xytext=(prev_end, i),
        arrowprops=dict(arrowstyle="->", color="black")
    )

# ================= LEGEND =================
ax.legend()

# ================= LAYOUT =================
plt.tight_layout()
plt.show()