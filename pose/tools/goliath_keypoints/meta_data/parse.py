import csv
import re
import json

# Load colors from the JSON files
color_dicts = {}
for filename in ["face.json", "body.json", "hand.json", "joints.json"]:
    with open(filename, "r") as file:
        data = json.load(file)
        color_dicts.update(data["colors"])

# Read data from the data.txt file
with open("data.txt", "r") as file:
    content = file.read()

# Extract the uuid and name using regex
pattern = r'- uuid: (\S+)\s+name: (\S+)'
matches = re.findall(pattern, content)

# Create a new list to store the data with the colors
data_with_colors = []

for index, match in enumerate(matches, start=1):
    uuid, name = match
    # Get the tuple of RGB values or a default value if not found
    color_tuple = tuple(color_dicts.get(uuid, [None, None, None]))
    data_with_colors.append([index, name, uuid, color_tuple])

# Write the data with colors to the CSV file
with open("parsed_data.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the headers
    writer.writerow(["keypoint_index", "keypoint_name", "uuid", "color"])
    
    # Write the data with colors
    writer.writerows(data_with_colors)
