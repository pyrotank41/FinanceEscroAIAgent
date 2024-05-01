import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

data_file_path = "eval_results/rag_eval_results.csv"
output_image_path = "media/rag_eval_table.png"

# Function to apply color based on value
def apply_color(value, max_val):
    
    if isinstance(value, str) or isinstance(value, str):
        return None
    
    # if value == max_val:
    #     return "Cyan"  # Highlight maximum values distinctly
    
    if value < 0.5:
        return "red"
    elif 0.5 <= value < 0.8:
        return "yellow"
    else:
        return "green"


# Read the CSV file
df = pd.read_csv(data_file_path)

# Sort the DataFrame based on the "Total Cost"
df = df.sort_values(by="total_cost")
# round the values
df = df.round(4)

# Identify the maximum values for each column
max_values = df.select_dtypes(include=[np.number]).max()
print(max_values)

# Rendering DataFrame as a Matplotlib Table and saving as an image
fig, ax = plt.subplots(figsize=(12, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
colWidths = [0.2] * len(df.columns)
colWidths[0] = 0.3  # Adjust the first column width
colWidths[-1] = 0.1  # Adjust the last column width

# Create table
the_table = table(ax, df,
                  loc='center', 
                  cellLoc='center',
                  colWidths=colWidths,
                )

# Apply colors
for (i, j), val in np.ndenumerate(df):
    # Skip the first row (headers) and column (index)
    if i != 0 and j != 0:
        header = df.columns[j]
        # Get the maximum value for the current column
        max_val = max_values[header]
        cell_color = apply_color(val, max_val)
        the_table.get_celld()[(i+1, j)].set_facecolor(cell_color)
        
    the_table.get_celld()[(i+1, j)].set_edgecolor('black')

the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1, 1.4)  # Adjust table scale


# Save the image
plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
plt.close()

