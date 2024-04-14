from datetime import date
import nsepy

data = nsepy.get_history(symbol='ADANIENT', start=date(2021, 3, 1), end=date(2023, 2, 27))
data.drop(["Symbol", "Series", "Prev Close", "VWAP", "Volume", "Deliverable Volume", "%Deliverble"], axis = 1, inplace=True)
data.to_csv('Adani.csv')