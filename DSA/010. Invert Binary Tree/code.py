# Invert Binary Tree
# Description: Invert the left and right subtrees of a binary tree.
# Example: Input tree → Inverted tree
# Constraints: Recursive or iterative

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invertTree(root):
    if root is None:
        return None

    # Swap left and right
    root.left, root.right = root.right, root.left

    # Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)

    return root
