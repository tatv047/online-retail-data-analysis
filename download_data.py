import os 
import requests 
import zipfile

# setup path
url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
download_dir = os.path.join('.','data','raw_data') # destination folder
zip_path = os.path.join(download_dir,'temp_data.zip')

# create the directory if it doesn't exist
os.makedirs(download_dir,exist_ok=True)

# download the file
response = requests.get(url)
with open(zip_path,"wb") as f:
    f.write(response.content)

print(f"Downloaded the zip file to: {zip_path}")

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

print(f"Extracted contents to: {download_dir}")

#  Clean up the .zip file
os.remove(zip_path)
print(f"Removed temporary zip file: {zip_path}")
