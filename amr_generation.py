import re
import os
import subprocess
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import amrlib
stog = amrlib.load_stog_model()

# Function to run the code for each sentence
def run_code(snt):
    output=""
    graphs = graphs = stog.parse_sents([snt])
    for graph in graphs:
        output+=graph
    return output

f = sys.argv[1]
# Read the input file
with open(f, "r") as file:
    lines = file.readlines()

# Parse the file and extract sentences
sentences = []
for i, line in enumerate(lines):
    if line.startswith("# ::snt"):
        sentence = re.search(r'# ::snt (.+)', line).group(1)
        sentences.append(sentence.strip())
        lines[i]=""

# Run code for each sentence and append the output
output_lines = []
for sentence in tqdm(sentences, total=len(sentences)):
    output = run_code(sentence)
    output_lines.append(output)

# Update the lines by removing the next two lines after each '# ::save-date' line
g = sys.argv[2]
remove_next_two = False
for i in range(len(lines)-1, -1, -1):
    if remove_next_two:
        lines.pop(i)
        remove_next_two = False
    elif lines[i].startswith("# ::save-date"):
        remove_next_two = True

# Append the output lines to the file after corresponding '# ::save-date' sentences
output_lines_index = 0
for i, line in enumerate(lines):
    if line.startswith("# ::save-date"):
        # lines[i] = line.rstrip()  # Remove trailing newline
        lines[i+1]=""
        lines[i+2]="\n"
        output_lines_index = min(output_lines_index, len(output_lines) - 1)
        output_line = output_lines[output_lines_index]
        lines.insert(i + 1, output_line)
        output_lines_index += 1

# Write the updated content to the file
with open(g, "w") as file:
    file.writelines(lines)