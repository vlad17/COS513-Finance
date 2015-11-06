""" file scraper """

import urllib
from os import getcwd
import sys


gdelt_site = 'http://data.gdeltproject.org/events/'
url_opener = urllib.URLopener()

year = sys.argv[1]
month = sys.argv[2]


MIN_NUM_EVENTS = 100



def download_file(filename):
    filepath = gdelt_site + filename

    try:
        url_opener.retrieve(filepath, filename)
        print 'downloaded ' + filename
    except:
        print 'no file ' + filepath


def split_by_day(filename):
    with open(filename, 'r') as data:
        day_buffer = []
        current_date = ''
        previous_date = current_date

        for line in data:
            try:
                current_date = line.split('\t')[1]
            except:
                pass
            if current_date != previous_date:
                
                with open(current_date + '.export.CSV.zip', 'w') as outf:
                    if len(day_buffer) >= MIN_NUM_EVENTS:
                        for line in day_buffer:
                            outf.write("%s" % line)

                    print('wrote {}'.format(current_date))

                    day_buffer = []

            day_buffer.append(line)

            previous_date = current_date

        
year_i = int(year)
month_i = int(month) 

# daily files
if year_i > 2013 or (year_i == 2013 and month_i > 3):
    for day in range(1,32):
        filename = year + str(month).zfill(2) + str(day).zfill(2) + '.export.CSV.zip'
        download_file(filename)

# monthly files
elif year_i > 2005: 
    # 2013 switches from monthly to daily starting April 1st
    if year_i == 2013:
        for m in range(0,4):
            filename = year + str(month).zfill(2) + '.zip'
            download_file(filename)
            split_by_day(filename)
    else:
        for m in range(0,13):
            filename = year + str(month).zfill(2) + '.zip'
            download_file(filename)
            split_by_day(filename)

elif year_i > 1978:
    filename = year + '.csv'
    download_file(filename)
    split_by_day(filename)




