import requests
from bs4 import BeautifulSoup
import re
import json

# web page to scrape
url = "https://en.wikipedia.org/wiki/Data_science"

# request page
r = requests.get(url)

# parse page contents
soup = BeautifulSoup(r.content, 'html.parser')

found_text_list = []

# finds any string containing 'data scientist', case insensitive
for x in (soup.find_all(string=re.compile('data scientist', flags=re.I))):
	# Adds the found string and its parent into list for json export
	found_text_list.append({'Parent' : str(x.parent), 'String': x})

# Prepare data for json export
jsondata = {'Data Scientist Occurences' : found_text_list}

# Prints the data in an easy to read format
print(json.dumps(jsondata, indent = 4))

# Writes to json file
with open('DataScientistJSON.json', 'w') as outputfile:
	json.dump(jsondata, outputfile)