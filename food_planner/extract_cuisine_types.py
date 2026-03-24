import pandas as pd
import os

# Load the CSV file into a DataFrame
RECIPE_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/recipes.csv')

df = pd.read_csv(RECIPE_DATA_PATH)

# Extract the 'cuisine_path' column
cuisine_path_column = df['cuisine_path']

# Split the values in the 'cuisine_path' column by a delimiter (e.g., '/')
# and flatten the list to get all unique recipe types
# Extract the first component of each path and collect unique values
first_components = cuisine_path_column.dropna().apply(lambda x: x.split('/')[1]).unique()

# Convert the result to a sorted list for better readability
unique_first_components = sorted(first_components)

# Convert the set to a sorted list for better readability
# unique_recipe_types = sorted(recipe_types)

# Print the unique recipe types
print("Unique Recipe Types:")
for recipe in unique_first_components:
    print(recipe) 
    
    