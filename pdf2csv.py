import pyperclip
import csv

headers = ('Year', 'Nonmonsoon', 'Monsoon', 'Annual-average', )
bad_headers = ('Nonmonsoon', 'Monsoon')
raw = pyperclip.paste().replace(',', '').replace('Annual average', 'Annual-average').replace(' ', '\n').split('\n')
raw_bc = raw[0:raw.index('62.32')]
rows = []
counter = -1
header = None
for elm in raw:
    if elm in headers:
        header = elm
        if not header in bad_headers:
            counter += 1
            rows.append([header])
    else:
        if not header in bad_headers:
            rows[counter].append(float(elm))
csv_dict = []
for i in range(len(rows[0])):
    csv_dict.append({'Yr': rows[0][i], 'BC': rows[1][i], 'OC': rows[2][i]})

#save to csv
    fields = ["Yr", "BC", "OC"]
    with open("test.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_dict)