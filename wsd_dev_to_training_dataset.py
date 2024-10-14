import json
import pandas as pd

# Load JSON files
with open('/content/2-dev-ground-truth.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

with open('/content/3-dev-sense-dictionary.json', 'r', encoding='utf-8') as f:
    dictionary = json.load(f)

with open('/content/1-dev-set.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Create a dictionary to map sense IDs to their meanings
sense_meanings = {sense['sense_id']: sense['definition'] for sense in dictionary}

# Create a list to store the data
data = []

# Iterate through the dataset
for item in dataset:
    sentence_id = item['sentence_id']
    sentence = item['sentence']
    for word in item['words']:
        word_id = word['word_id']
        word_text = word['word']
        senses = word['senses']

        # Check if each sense is present in the ground truth
        for sense_id in senses:
            # Set label based on presence in ground truth
            label = 1 if any(sense_id == word['target_sense'] for word in ground_truth[sentence_id - 1]['words']) else 0

            # Get the meaning of the sense from the dictionary
            meaning = sense_meanings.get(sense_id, '')

            # Construct the tokenized sentence with <token> marking the word of interest
            tokenized_sentence = "[SEP] " + sentence.replace(word_text, '<\\token>' + word_text + '<token>') + " [CLS] " + meaning + " [SEP]"


            # Append data to the list
            data.append({'sentence': sentence, 'word': word_text, 'label': label, 'meaning': meaning, 'tokenized_sentence': tokenized_sentence})

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())

# Save DataFrame to CSV
df.to_csv('dataset.csv', index=False, encoding='utf-8-sig')
