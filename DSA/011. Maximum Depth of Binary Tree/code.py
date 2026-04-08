# Maximum Depth of Binary Tree
# Description: Find the maximum depth of a binary tree.
# Example: [3,9,20,null,null,15,7] → 3
# Constraints: DFS or BFS

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxDepth(root):
    if root is None:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    return 1 + max(left_depth, right_depth)
