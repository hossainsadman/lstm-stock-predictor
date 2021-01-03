import os
from dotenv import load_dotenv
import urllib.request
import json
import pandas
import datetime

load_dotenv()
ALPHA_API_KEY = os.getenv('ALPHA_API_KEY')
TICKER = 'QCOM'

url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s" %(TICKER, ALPHA_API_KEY)
data_file = '%s.csv' %TICKER

if not os.path.exists(data_file):
    data = json.loads(urllib.request.urlopen(url).read().decode())
    data = data['Time Series (Daily)']

    df = pandas.DataFrame(columns=['Date','Low','High','Close','Open'])
    for i, j in data.items():
        date = datetime.datetime.strptime(i, '%Y-%m-%d')
        data_row = [date.date(), float(j['3. low']), float(j['2. high']), float(j['4. close']), float(j['1. open'])]
        df.loc[-1,:] = data_row
        df.index += 1
    df.to_csv(data_file, index=False)
