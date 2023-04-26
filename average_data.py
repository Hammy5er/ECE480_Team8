import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def write_csv(data, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def average_first_80_percent(data):
    header = data[0]
    data = data[1:]
    averaged_data = [header]

    boundary = int(0.8 * len(data))

    for i in range(0, boundary - 1, 2):
        avg_line = [(float(data[i][k]) + float(data[i + 1][k])) / 2 for k in range(len(data[0]))]
        averaged_data.append(avg_line)

    averaged_data.extend(data[boundary:])

    return averaged_data

input_csv_file = 'output_data.csv'  # Replace with your input file name
output_csv_file = 'average_robot_data.csv'  # Output file with the averaged data

data = read_csv(input_csv_file)
averaged_data = average_first_80_percent(data)
write_csv(averaged_data, output_csv_file)

print(f"Averaged first 80% of data from {input_csv_file} and saved to {output_csv_file}.")
