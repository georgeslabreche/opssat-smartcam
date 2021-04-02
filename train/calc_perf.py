#!/usr/bin/python

import sys
import csv

# Check that a valid number of arguments is given.
if len(sys.argv) != 2:
    print('Invalid arguments. Usage e.g.: python3 calc_perf.py <model_name>')
    exit(1)

# The name of the model that is going to be trained.
model_name = sys.argv[1]

# Path of the CSV log file.
CSV_FILENAME = 'repo/' + model_name + '/data/classification/confidences.csv'

# Calculate prediction accuracy.
results = {}

# Read the CSV log file
with open(CSV_FILENAME, mode='r') as csv_file:

    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:

        # Get the predicted label as well as the expectd label
        predicted_label = row['predicted_label']
        expected_label = row['expected_label']

        # Count number of correct predictions and as well as total predictions made.
        if predicted_label == expected_label:

            if expected_label in results:
                results[expected_label]['correct'] += 1
                results[expected_label]['total'] += 1

                # Update the accuracy:
                results[expected_label]['accuracy'] = results[expected_label]['correct'] / results[expected_label]['total']

            else:
                # Create the label entry if this is the first time that the label is encounterd.
                results[expected_label] = {
                    'correct': 1,
                    'total': 1,
                    'accuracy': 0.0
                }

        # for incorrect predictions, just update the total count.
        else:
            if expected_label in results:
                results[expected_label]['total'] += 1

                # Update the accuracy:
                results[expected_label]['accuracy'] = results[expected_label]['correct'] / results[expected_label]['total']

            else:
                # Create the label entry if this is the first time that the label is encounterd.
                results[expected_label] = {
                    'correct': 0,
                    'total': 1,
                    'accuracy': 0.0
                }

total_correct = 0
total_predictions = 0

print("\nACCURACY REPORT:\n")
for key in results:
    print(key + ':\n  ' + str(float(results[key]['correct']) / float(results[key]['total'])))
    print('  ' + str(results[key]['correct']) + '/' + str(results[key]['total']) + '\n')
    total_correct += results[key]['correct']
    total_predictions += results[key]['total']


print('average:\n  ' + str(float(total_correct) / float(total_predictions)))
print('  ' + str(total_correct) + '/' + str(total_predictions) + '\n')