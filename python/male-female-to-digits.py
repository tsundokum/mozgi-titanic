import csv

with open('../data/train.csv','rb') as input_file:
    reader = csv.DictReader(input_file)

    with open('../data/train_male_female_to_digits.csv', 'wb') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            new_row = row
            new_row['sex'] = 0 if row['sex'] == "female" else 1

            writer.writerow(new_row)

