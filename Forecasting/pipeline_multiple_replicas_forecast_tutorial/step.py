import pandas as pd

# take a dataframe output of the house price model, and reformat the `dense_2`
# column as `output`
def wallaroo_json(data):
    return pd.DataFrame({"forecast": [data['forecast']]})