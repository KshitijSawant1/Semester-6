# Contains Duplicate
# Description: Determine if any value appears at least twice in the array.
# Example: [1,2,3,1] → true
# Constraints: Use efficient time complexity

def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


if __name__ == "__main__":
    nums = [1, 2, 3, 1]
    print(containsDuplicate(nums))  
    
# Output: True
