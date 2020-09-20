# Ultimate Challenge Functions Library

# __Modules__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def import_logins(data):
    """
    The "logins.json" file is not in a valid format for json import.
    This function loads in a generic file that is written in a Python
    dictionary format and returns it as a Pandas DataFrame.
    """

    with open(data) as logins_file:
        logins_raw = logins_file.read()
        logins_dict = eval(logins_raw)
        logins_df = pd.DataFrame(logins_dict)
    return logins_df


def define_bins(data):
    """
    Generates a list of 15 minute time intervals to iterate over for the
    bin_datetime function.
    """
    # delta = pd.to_timedelta(data)
    fifteen_minutes = pd.Timedelta(minutes=15)
    steps = [min(data)]
    idx = 0
    while steps[-1] < data.iloc[-1]:
        steps.append(steps[idx] + fifteen_minutes)
        idx += 1
    return steps


def bin_datetimes(data):
    """
    Transforms all values in the 'login_time' column of logins_df
    to a binned datetime object in 15 minute intervals.
    - Requires a dataframe with the 'login_time' column of datetimes.
    - This happens inplace and specfically on the dataframe named "logins_df"
    """
    bins = define_bins(data)
    rv_dict = {'login_bin': [], 'value_count': []}
    for bin in bins:
        if bin == bins[-1]:
            break
        count = sum([(data >= bin) & (data < bins[bins.index(bin)+1])])
        rv_dict['login_bin'].append(bin)
        rv_dict['value_count'].append(sum(count))
    rv_df = pd.DataFrame(rv_dict)
    return rv_df


def batch_data(data, num_batches=10):
    """
    Splits a series into batches of a specified size.
    - I needed this to run split up the run-time when using
    the bin_datetimes function.
    """
    batch_size = round(len(data)/num_batches)
    batches = []
    size = len(data)
    steps = np.arange(0, size, batch_size).tolist()
    idx = 0


    while idx < len(steps):
        if steps[idx] == steps[-1]:
            break
        batch_df = data.iloc[steps[idx]:steps[idx+1]]
        batches.append(batch_df)
        idx += 1

    print('Batch Size: ', batch_size,
    '\nBatches: ', num_batches,
    '\nOriginal: ', size)
    return batches

def process_batch(batched_data, function):
    """
    Automation on batched data
    - Used to automate the for loop with logins_df_batched
    through bin_datetimes.
    - Returns a generator that can be used to write processed data.
    """
    for batch in batched_data:
        yield function(batch)

def plot_bins(data, path=''):
    """
    Plots the logins dataframe in a bar chart show trends
    by day of user logins.
    - data -> DataFrame
    - path -> Path of save file for plt.savefig.
    """

    plt.figure(figsize=(20,10))
    plt.title('User Login Activity')
    plt.bar(x=data['login_bin'],
            height=data['value_count'],
            color='black')

    start = min(data['login_bin'])
    end = max(data['login_bin'])
    ticks = pd.date_range(start=start, end=end, freq='24H')
    plt.xticks(ticks=ticks, rotation=90)
    plt.savefig(path+'user_login_activity.png', transparent=True)

def one_hot(dataframe, column):
    """
    Encodes classified data into numerical values and concatonates it back
    into the dataframe.
    """
    dummies = pd.get_dummies(dataframe[column])
    dataframe_encoded = pd.concat([dataframe, dummies], axis=1)
    dataframe_dropped_columns = dataframe_encoded.drop(column, axis=1)
    return dataframe_dropped_columns


    return dataframe

def scatter_explore(x, ys, output_dir=''):
    """
    Iterates through an array of y values and compares them with a static x value.
    Output files are then saved to the output_dir
    x -> array
    y -> iterable of arrays
    """
    for y in tqdm(ys):
        plt.figure(figsize=(8,6))
        title = x.name + ' vs ' + y.name
        plt.title(title.title())
        plt.scatter(y, x, color='black')
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.savefig(output_dir+title.replace(' ', '_')+'.png', transparent=True)
