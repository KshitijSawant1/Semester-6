# Move Zeroes
# Description: Move all zeros to the end while maintaining the order of non-zero elements.
# Example: [0,1,0,3,12] → [1,3,12,0,0]
# Constraints: In-place operation

def moveZeroes(nums):
    insert_pos = 0

    # Move non-zero elements forward
    for num in nums:
        if num != 0:
            nums[insert_pos] = num
            insert_pos += 1

    # Fill remaining positions with zero
    for i in range(insert_pos, len(nums)):
        nums[i] = 0


if __name__ == "__main__":
    nums = [0, 1, 0, 3, 12]
    moveZeroes(nums)
    print(nums)  

# Output: [1, 3, 12, 0, 0]
