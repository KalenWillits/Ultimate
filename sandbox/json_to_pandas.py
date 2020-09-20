# Sandbox file
# Testing .json load ins to pandas.

import numpy as np
import pandas as pd
import json
cd_data = 'data/'
!pwd

def import_logins(data):
    
    """
    The "logins.json" file is not in a valid format for json import.
    This function loads in a generic file that is written in a Python dictionary
    format and returns it as a Pandas DataFrame.
    """

    with open(cd_data + data) as logins_file:
        logins_raw = logins_file.read()
        logins_dict = eval(logins_raw)
        logins_df = pd.DataFrame(logins_dict)
    return logins_df
