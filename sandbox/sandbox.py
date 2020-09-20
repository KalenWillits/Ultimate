# sandbox.py

logins_df.info()

unique = logins_df[logins_df.columns[0]].unique()
min(unique)
np.linspace(pd.Timestamp(min(unique)).value, pd.Timestamp(max(unique)).value, num=15)
# _______________________________________________________________________
def bin_datetimes(logins_df):

    """

    """
    start = pd.Timestamp(min(logins_df['login_time'])).value
    end = pd.Timestamp(max(logins_df['login_time')).value
    fifteen_days = 60*60*15   # seconds*minutes*hours*days
    bins = np.linspace(start, end, fifteen_days)
    bins_dt = pd.to_datetime(bins)

    def transform_dt(dt, bins, idx=0):
        """
        Nested function within "bin_datetimes"
        Takes a datetime value from logins_df and 'bins' it
        """
        if (dt >= bins[idx]) and (dt < bins[idx+1]):
            dt = bins[idx]
        return dt

    idx = 0
    while idx < len(bins):
        logins_df['login_time'].apply(transform_dt)
        idx += 1
    return logins_df


    # The transform function takes too long. I will need to iterate through
    # pandas filters instead.




    def to_num(dt):
        return dt.dt.total_seconds().div(60).astype(int)


    data = logins_df

    col = data.columns[0]

    start_minute = min(data[col].dt.minute)
    start_hour = min(data[col].dt.hour)
    start_day = min(data[col].dt.day)

    stop_minute = max(data[col].dt.minute)
    stop_hour = max(data[col].dt.hour)
    stop_day = max(data[col].dt.day)


    step = 15
    # scale = 1000000
    minutes_bucket = np.arange(start_minute, stop_minute, step)
    hours_bucket = np.arange(start_hour, stop_hour)
    days_bucket = np.arange(start_day, stop_day)

    times_matrix = []
        for day in days_bucket:
            for hour in hours_bucket:
                for minute in minutes_bucket:
                    date = pd.to_datetime(0)
                    date.day += day
                    date.hour += hour
                    date.minute += minute
                    times_matrix.append(date)



start = pd.to_timedelta(min(logins_df['login_time']).value)
end = pd.to_timedelta(max(logins_df['login_time']).value)
freq = '15m'
start
pd.timedelta_range(start=start, end=end, freq=freq)

#_____________________________________`


# Need to find an array of date times to compare with.

    logins_df['login_time']+fifteen_minutes
    bot = (logins_df['login_time'] >= min(logins_df['login_time']))
    top = (logins_df['login_time'] < min(logins_df['login_time'])+fifteen_minutes)


logins_df[bot & top]


logins_binned.describe()
)





import datetime

def bin_datetimes(data):
    for date in data:
        if data.time() < datetime.time():
            return "00:00-05:59"


df["Bin"] = df["TimeStamp"].apply(bin_f)
grouped = df.groupby("Bin")
grouped['X'].agg(np.std)






pd.Timedelta(logins_df['login_time'].values[0])

def define_bins(data):
    fifteen_minutes = pd.Timedelta(minutes=15)
    steps = [min(data)]
    for step in tqdm(steps):
        try:
            steps.append(step + fifteen_minutes)
        except:
            pass
    return steps

bins = define_bins(logins_df['login_time'])
bins

for bin in bins:
    print(bins.index(bin))


df = pd.DataFrame({'col1':list(range(1,100))})

df[df['col1'] > 10] = 5


binned = pd.cut(df['col1'], 100)


binned.head()



logins_df = logins_df.head(1000)
logins_df.info()


round(5.5)

def batch_data(data, num_batches=10):
    """
    """
    batch_size = round(len(data)/num_batches)
    batches = []
    for batch in range(num_batches):
        batch_df = batch.head(batch_size)
        batches.append(batch_df)
    print('Batch Size: ', batch_size,
    '\nBatches: ', num_bat,
    '\nOriginal: ', len(data))
    return batches

    x = pd.DataFrame({'col':range(100000)})

for x in batch_data(x):
    print(x.head())

    x.iloc[123:900]


    np.arange(0, 100, 10)

v = round(len(logins_df)/10)
print(v)

steps = np.arange(0, v, round(v/10)).tolist()
steps

steps.index(0)
x.iloc[0:steps[steps.index(0)+1]]



data = pd.DataFrame({'col':range(100)})
data = logins_df['login_time']
bins

from library import *
"""
Transforms all values in the 'login_time' column of logins_df
to a binned datetime object in 15 minute intervals.
- Requires a dataframe with the 'login_time' column of datetimes.
- This happens inplace and specfically on the dataframe named "logins_df"
"""
bins = define_bins(data)
rv_dict = {'login_bin': [], 'value_count': []}
for bin in bins:
    bot = (data >= bin)
    if bin = bins[-1]:
        break
        top = (data < bins[bins.index(bin)+1])
    count = sum(data[bot & top])
    rv_dict['login_bin'].append(bin)
    rv_dict['value_count'].append(sum(count))
    rv_df = pd.DataFrame(rv_dict)
return rv_df


fifteen_minutes = pd.Timedelta(minutes=15)
steps = np.arange(min(data), max(data), )
for step in steps:
    try:
        steps.append(step + fifteen_minutes)
    except:
        continue

fifteen_minutes
data = logins_to_time
data
min(data)
"""
Generates a list of 15 minute time intervals to interate over for the
bin_datetime function.
"""
# delta = pd.to_timedelta(data)
logins_to_time + pd.Timedelta(minutes=15)
pd.to_datetime(logins_df[logins_df.columns[0]]).dt.minute
import datetime
fifteen_minutes = datetime.timedelta(minutes=15)
fifteen_minutes + fifteen_minutes
steps
steps = [min(data)]
idx = 0
while steps[idx] < data.iloc[-1]:
    steps.append(steps.astype(np.timedelta64) + fifteen_minutes)
    idx += 1
return steps
cd_data = 'data/'
import pandas as pd; df =  pd.read_csv(cd_data+'logins_binned.csv')


df.head(20)

data = logins_to_time
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


s = ' Visualizing Binned Login Data'.title().strip()
ss = '#'+s.replace(' ', '-')

print(s)
print(ss)

pd.read_json(cd_data+'ultimate_data_challenge.json')





rider_df.columns




rider_df.isnull().sum()

#  Inspecting the dataframe.
for col in rider_df.columns:
    print(rider_df[col].head(20))
arbd = np.mean(rider_df['avg_rating_of_driver'])
plt.hist(rider_df['avg_rating_by_driver'].fillna(arbd)); plt.hist(rider_df['avg_rating_by_driver'])

# If the mean is a bad value, how much will it affect the experiment?
(rider_df['avg_rating_by_driver'].fillna(arbd).sum() - rider_df['avg_rating_by_driver'].sum())/rider_df['avg_rating_by_driver'].sum()
# Not by much, this is not worth bootstrapping.

# The avg_rating_of_driver, phone, and avg_rating_by_driver contains null values.

# After playing in the sandbox for a while, I feel fairly good about using the mean of avg_rating_by_driver to replace nans in that column.
rider_df.describe()

(rider_df['avg_rating_of_driver'].fillna(arbd).sum() - rider_df['avg_rating_of_driver'].sum())/rider_df['avg_rating_by_driver'].sum()

plt.hist(rider_df['avg_rating_of_driver'])

plt.hist(np.random.inverse_exponential(rider_df['avg_rating_by_driver']))

rider_df.isnull().sum()
seed = 1111
data = rider_df['avg_rating_of_driver']
np.random.seed(seed)
nulls = data[data.isnull()]
notnulls = data[data.notnull()]
idx = 0
for entry in data:
    if entry == np.NaN:
        data.iloc[idx] = notnulls[np.random.randint(len(notnulls))]
rider_df['city']
rider_df.columns
rider_encoded_df.info()


def columns_to_datetimes(dataframe, columns):
    """
    """
    for column in columns:
        datetime_series = pd.to_datetime(dataframe[column])
        dataframe_encoded = pd.concat([dataframe, datetime_series], axis=1)
        dataframe_dropped_columns = dataframe_encoded.drop(column, axis=1)
    return dataframe
