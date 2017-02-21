import quandl
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Openen van dataset
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
forecast_out = int(math.ceil(0.001*len(df)))
print("Aantal dagen die vooruit voorspeld worden ", forecast_out)

# Uiteindelijke label bepaling
df['label'] = df[forecast_col].shift(-forecast_out)

# Dropping van alle NA's, dus wanneer er geen data beschikbaar is.
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

# Preprocessing van data op een scale van -1 tot 1
X = preprocessing.scale(X)
y = np.array(df['label'])

# Cross validation, X en Y blijven gekoppeld maar worden random door elkaar gegooit
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Lineair regression word geinitialiseerd, n_jobs=-1 betekend max aantal threads die je cpu aan kan.
clf = LinearRegression()

# Fitting is het trainen van je Linear Regression
clf.fit(X_train, y_train)

# Accuracy berekening, hier word je testing set voor gebruikt.
accuracy = clf.score(X_test, y_test)
print(accuracy)