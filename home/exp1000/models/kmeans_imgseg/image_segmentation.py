#!/usr/bin/python3
import json
import time

# The JSON results object.
results = {
    'cloudy_0_25': 1,
    'cloudy_26_50': 0,
    'cloudy_51_75': 0,
    'cloudy_76_100': 0
}

# Convert JSON to String.
json_str = json.dumps(results)
 
# Return the results as a JSON String.
print(json_str)