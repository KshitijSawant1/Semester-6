# Plus One
# Description: Add one to a number represented as an array of digits.
# Example: [1,2,9] → [1,3,0]
# Constraints: Handle carry
def plusOne(digits):
    n = len(digits)

    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0

    return [1] + digits


# Example usage
if __name__ == "__main__":
    digits = [1, 2, 9]
    print(plusOne(digits))  
    
# Output: [1, 3, 0]
