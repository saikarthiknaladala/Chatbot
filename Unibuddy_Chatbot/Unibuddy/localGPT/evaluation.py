import csv
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

def reformat_csv(file_path):
    data = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if present
        for row in reader:
            if len(row) >= 2:
                question, correct_answer = row[:2]  # Assuming first two columns are question and correct_answer
                data.append((question.strip(), correct_answer.strip()))
    return data


def evaluate_bleu(evaluation_data, predicted_answers):
    bleu_scores = []
    # Iterate over evaluation data
    i=0
    for prompt, reference_answer in evaluation_data:
        # Calculate BLEU score
        bleu_score = sentence_bleu([reference_answer.split()], predicted_answers[i].split())
        bleu_scores.append(bleu_score)
        i+=1

    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

    return avg_bleu_score

# Example usage:
csv_file_path = 'LLM Evaluation.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file_path)
evaluation_data = reformat_csv(csv_file_path)

predicted_answers = list(data['generated_answer'])
print(predicted_answers)
# Perform evaluation
avg_bleu_score = evaluate_bleu(evaluation_data, predicted_answers)

# Print or log BLEU score
print(f"Average BLEU Score: {avg_bleu_score}")