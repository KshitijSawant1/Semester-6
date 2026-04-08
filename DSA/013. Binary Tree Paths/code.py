# Binary Tree Paths
# Description: Return all root-to-leaf paths in a binary tree.
# Example: ["1→2→5", "1→3"]
# Constraints: Backtracking

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def binaryTreePaths(root):
    result = []

    def backtrack(node, path):
        if not node:
            return

        path.append(str(node.val))

        # If leaf node, add path to result
        if not node.left and not node.right:
            result.append("->".join(path))
        else:
            backtrack(node.left, path)
            backtrack(node.right, path)

        path.pop()  # backtrack

    backtrack(root, [])
    return result
