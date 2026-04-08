# Path Sum
# Description: Check if a tree has a root-to-leaf path equal to the target sum.
# Example: target = 22 → true
# Constraints: Tree traversal

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def hasPathSum(root, targetSum):
    if root is None:
        return False

    # If leaf node, check remaining sum
    if not root.left and not root.right:
        return targetSum == root.val

    # Recurse on left and right subtrees
    remaining = targetSum - root.val
    return (
        hasPathSum(root.left, remaining) or
        hasPathSum(root.right, remaining)
    )
