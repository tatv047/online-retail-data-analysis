import pandas as pd 

# load the data
print('Loading the raw data')
try: 
    df = pd.read_excel('./data/raw_data/Online Retail.xlsx')
    print('Raw data loaded sucessfully')
except:
    print('Raw Data Loading error')

print('='*60)

cleaned_df = df.copy() # making a copy

print("Cleaning the 'InvoiceNo' column...")

cleaned_df['InvoiceNo'] = cleaned_df['InvoiceNo'].astype('str')
mask = (
    cleaned_df['InvoiceNo'].str.match('^\\d{6}') == True
)
cleaned_df = cleaned_df[mask]

print("'InvoiceNo' cleaned successfully")
print(f'%cent size of the original data: {len(cleaned_df)/len(df)}')
print('='*60)

print("Cleaning 'StockCode' column...")
cleaned_df['StockCode'] = cleaned_df['StockCode'].astype('str')

mask = (
    (cleaned_df['StockCode'].str.match('^\\d{5}$') == True)              # exactly 5 digits
    | (cleaned_df['StockCode'].str.match('^\\d{5}[a-zA-Z]+$') == True)   # 5 digits long followed by letters
    | (cleaned_df['StockCode'].str.match('^PADS$') == True)              
)

cleaned_df = cleaned_df[mask]

print("'StockCode' cleaned successfully")
print(f'%cent size of the original data: {len(cleaned_df)/len(df)}')
print('='*60)

print("Cleaning 'CustomerID' column...")
cleaned_df.dropna(subset = ['CustomerID'],inplace= True)

print("'CustomerID' cleaned successfully")
print(f'%cent size of the original data: {len(cleaned_df)/len(df)}')
print('='*60)

print("Cleaning 'UnitPrice' column...")
cleaned_df = cleaned_df[cleaned_df['UnitPrice'] > 0]

print("'UnitPrice' cleaned successfully")
print(f'%cent size of the original data: {len(cleaned_df)/len(df)}')
print('='*60)

print("Cleaning 'Quantity' column...")
cleaned_df = cleaned_df[cleaned_df['Quantity'] > 0]

print("'Quantity' cleaned successfully")
print(f'%cent size of the original data: {len(cleaned_df)/len(df)}')
print('='*60)

print('Data Cleaning completed')
print(f'Original data size: {df.shape}')
print(f'Cleaned datasize: {cleaned_df.shape}')

print('='*60)
print('Saving the cleaned data...')
try:
    cleaned_df.to_csv('./data/cleaned_data/online_retail_df.csv',index=False)
    print('Saved succesfully !!')
except:
    print('cleaned data saving unsuccessful !!!')