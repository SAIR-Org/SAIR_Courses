from config import *

#Load data
df = pd.read_csv("Data/Power Plant Data.csv")
X = df.drop('PE', axis=1)
Y = df['PE']
print(f'Dataset shape: {df.shape}')
print(f'Dataset features: {df.columns}')
df.head(15)

# Info and Statistical Summary
display(df.describe())

# Check Missing Values
print(f'⚙️ Missing Values check: \n{df.isnull().sum()}')

# Data types check
print(f'⚙️ Data types: \n{df.dtypes}')


