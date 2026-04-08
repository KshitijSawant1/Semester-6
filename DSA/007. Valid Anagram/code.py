# Valid Anagram
# Description: Check whether two strings are anagrams of each other.
# Example: "anagram", "nagaram" → true
# Constraints: Same character frequency

def isAnagram(s, t):
    if len(s) != len(t):
        return False

    count = {}

    for ch in s:
        count[ch] = count.get(ch, 0) + 1

    for ch in t:
        if ch not in count:
            return False
        count[ch] -= 1
        if count[ch] < 0:
            return False

    return True


if __name__ == "__main__":
    s = "anagram"
    t = "nagaram"
    print(isAnagram(s, t))  
    
# Output: True
