import pandas as pd

# takes the JSON output
# column as `output`
def wallaroo_json(data):
    return pd.DataFrame({"forecast": [data['forecast']]})