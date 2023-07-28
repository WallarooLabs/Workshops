import pandas as pd

# takes the JSON output
# column as `output`
def wallaroo_json(data: pd.DataFrame):
    # Add in code that averages forecast elements
    return [
             data,
             {'average_rentals' : 1500}
           ]