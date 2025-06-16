# Load the text file
with open('data/A.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Replace multiple spaces/newlines/tabs by single space for clean splitting
import re
cleaned_text = re.sub(r'\s+', ' ', text)

# Split the text into a list of words
words_list = cleaned_text.split(' ')

# Remove empty strings if any (in case of leading/trailing spaces)
words_list = [word for word in words_list if word]

# Example: print first 50 words
print(words_list)
# Join the list of words back into a single string
joined_text = ' '.join(words_list)
# print(joined_text)

# Save the cleaned text back to a file
with open('data/A_cleaned.txt', 'w', encoding='utf-8') as file:
    file.write(joined_text)