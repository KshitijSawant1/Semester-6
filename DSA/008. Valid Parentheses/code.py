# Valid Parentheses
# Description: Determine if the parentheses string is valid.
# Example: "()[]" → true
# Constraints: Use stack

def isValid(s):
    stack = []
    mapping = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    for ch in s:
        if ch in mapping.values():
            stack.append(ch)
        elif ch in mapping:
            if not stack or stack.pop() != mapping[ch]:
                return False
        else:
            return False  # invalid character

    return len(stack) == 0


# Example usage
if __name__ == "__main__":
    s = "()[]"
    print(isValid(s))  # Output: True
