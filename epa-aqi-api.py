import json
import requests
import math
 
#Get api credentials
f = open('secrets.json')
data = json.load(f)
email = data['email']
api_key = data['epa-key']
f.close()

#define scope
east = [
    ["12", "073", "125", "0012", "0001", "Florida, Tallahassee"],
    ["13", "121", "033", "0032", "0001", "Georgia, Atlanta"],
    ["45", "079", "045", "0001", "0001", "South Carolina, Columbia"],
    ["37", "183", "029", "0017", "0001", "North Carolina, Raleigh"],
    ["51", "087", "159", "0014", "0001", "Virginia, Richmond"],
    ["24", "003", "001", "0014", "0001", "Maryland, Annapolis"],
    ["10", "001", "003", "0002", "0001", "Delaware, Dover"],
    ["34", "021", "033", "0005", "0001", "New Jersey, Trenton"],
    ["42", "043", "005", "0102", "0001", "Pennsylvania, Harrisburg"],
    ["36", "001", "003", "0005", "0001", "New York, Albany"],
    ["44", "007", "009", "0012", "0001", "Rhode Island, Providence"],
    ["09", "003", "015", "0013", "0001", "Connecticut, Hartford"],
    ["25", "025", "001", "0002", "0001", "Massachusetts, Boston"]
    ]

west = [
    ["53", "067", "023", "0013", "0001", "Washington, Olympia"],
    ["41", "047", "037", "0040", "0001", "Oregon, Salem"],
    ["06", "067", "049", "0006", "0001", "California, Sacramento"]
]

east_data = {"1990-1995": [0, 0], "1995-2000": [0, 0], "2000-2010": [0, 0]}
west_data = {"1990-1995": [0, 0], "1995-2000": [0, 0], "2000-2010": [0, 0]}

#request data
def get_dates(year):
    year = str(year)
    return (year + "0601", year + "0831")

def construct_url(state, county, year, site):
    by, ey = get_dates(year)
    return "https://aqs.epa.gov/data/api/sampleData/bySite?email=" + email + "&key=" + api_key + "&param=88101&bdate=" + by + "&edate=" + ey + "&state=" + state + "&county=" + county + "&site=" + site

for obj in west:
    for year in range(1990, 2011):
        state, urban, rural, u_site, r_site, meta = obj
        url = construct_url(state, urban, year, u_site)
        response = requests.get(url).json()
        data = response["Data"]
        for datum in data:
            pm = datum["sample_measurement"]
            if pm is not None and not math.isnan(pm) and (type(pm) == type(3) or type(pm) == type(1.1)):
                if year >= 2000:
                    key = "2000-2010"
                elif year < 1995:
                    key = "1990-1995"
                else:
                    key = "1995-2000"
                west_data[key][0] += pm
                west_data[key][1] += 1

print(west_data)

#units Micrograms PM2.5/cubic meter (LC)
#output:
#east {'1990-1995': [0, 0], '1995-2000': [4070.7000000000007, 207, 19.665217391304353], '2000-2010': [57521.39999999988, 3744, 15.363621794871761]}
#west {'1990-1995': [0, 0], '1995-2000': [560.8000000000001, 79, 7.09873417721519], '2000-2010': [4573.4, 626, 7.305750798722044]}
#east change: 2.770, 2.164
#west change: 1, 1.029
# 1 fire emoji = 7.0987 ug PM2.5/cubic meter/day (decadal averege)