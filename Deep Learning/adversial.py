import math

def minimax(depth, nodeIndex, isMax, values, maxDepth):
    if depth == maxDepth:
        return values[nodeIndex]

    if isMax:
        left = minimax(depth + 1, nodeIndex * 2, False, values, maxDepth)
        right = minimax(depth + 1, nodeIndex * 2 + 1, False, values, maxDepth)
        return max(left, right)
    else:
        left = minimax(depth + 1, nodeIndex * 2, True, values, maxDepth)
        right = minimax(depth + 1, nodeIndex * 2 + 1, True, values, maxDepth)
        return min(left, right)


n = int(input("Enter number of leaf nodes (must be power of 2): "))

values = []
print("Enter values of leaf nodes:")

for i in range(n):
    val = int(input(f"Value {i+1}: "))
    values.append(val)


depth = int(math.log2(n))


result = minimax(0, 0, True, values, depth)

print("\nOptimal value (Minimax):", result)