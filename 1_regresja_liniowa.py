import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../13012024_zajęcie/weight-height.csv')
print (df.head(3)) #wyświetl 3 pierwsze wiersze
print (df.Gender.value_counts()) #policz
df.Height *=2.54
df.Weight /= 2.2
print(f' po zmianie jednostek\n {df.head(3)}')

sns.histplot(df.Weight)
sns.histplot(df.query( "Gender =='Male'").Weight)
sns.histplot(df.query( "Gender =='Female'").Weight)

plt.show()

#zmiana płci na liczby

df = pd.get_dummies(df)
print(df.head())

#usunięcie kolumny 'male'

del (df['Gender_Male'])
print(df.head())
# dane wejściowe niezależne - height, gender, zalezne - weight

model = LinearRegression()
model.fit(df[['Height', 'Gender_Female']], df['Weight'])
print(f' Współczynnik kierunkowy: {model.coef_}, \nwyraz wolny: {model.intercept_}')
print(f' wzor: Height * {model.coef_[0]} +Gender * {model.coef_[1]} + {model.intercept_}')