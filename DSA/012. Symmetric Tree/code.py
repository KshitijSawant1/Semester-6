# Symmetric Tree
# Description: Check whether a binary tree is a mirror of itself.
# Example: [1,2,2,3,4,4,3] → true
# Constraints: Recursive comparison

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def isSymmetric(root):
    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (
            t1.val == t2.val and
            isMirror(t1.left, t2.right) and
            isMirror(t1.right, t2.left)
        )

    return isMirror(root.left, root.right) if root else True
