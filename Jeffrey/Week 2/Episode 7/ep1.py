import quandl, math, pickle, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# Style voor grafieken
style.use('ggplot')
# Openen van dataset
df = quandl.get("WIKI/GOOGL")


# De dataset vullen met alleen deze Headers
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Highlow percentage berekenen
df['HL_PCT'] =  (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

# Percentage verschil bepalen
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Alleen belangrijke dingen terug plaatsen in je dataframe
df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]

#
forecast_col = 'Adj. Close'

# Alle NaN in de dataset replacen met -99999, dit zorgt ervoor
# Dat deze worden gezien als outliers in niet meegenomen worden binnen
# De lineaire regressie.
df.fillna(-99999, inplace=True)

# Aantal dagen vooruit die je gaat voorspellen
forecast_out = int(math.ceil(0.01*len(df)))
print("Aantal dagen die vooruit voorspeld worden ", forecast_out)

# Uiteindelijke label bepaling
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

# Cross validation, X en Y blijven gekoppeld maar worden random door elkaar gegooit
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Lineair regression word geinitialiseerd, n_jobs=-1 betekend max aantal threads die je cpu aan kan.
# clf = LinearRegression(n_jobs=-1)
#
# # Fitting is het trainen van je Linear Regression
# clf.fit(X_train, y_train)
#
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')

clf = pickle.load(pickle_in)



# Accuracy berekening, hier word je testing set voor gebruikt.
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
