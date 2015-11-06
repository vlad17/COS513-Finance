""" file scraper """

import urllib
from os import getcwd
import sys
import zipfile
import os.path


gdelt_site = 'http://data.gdeltproject.org/events/'
url_opener = urllib.URLopener()

year = sys.argv[1]
month = sys.argv[2]


MIN_NUM_EVENTS = 100

outdir = '../raw-data/'



def download_file(filename):
    filepath = gdelt_site + filename

    if os.path.isfile(filename):
        print('{} already downloaded'.format(filename)) 
        return

    try:
        url_opener.retrieve(filepath, filename)
        print('downloaded ' + filename)
    except:
        print('no file ' + filepath)


def download_and_extract(filename):
    download_file(filename)
    unzip_filename = year + str(month).zfill(2) + '.csv'
    if not os.path.isfile(unzip_filename):
        fh = open(filename, 'rb')
        z = zipfile.ZipFile(fh)
        z.extractall()
    split_by_day(unzip_filename)

    os.remove(filename)
    os.remove(unzip_filename)


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

            if current_date != previous_date and previous_date != '':
                with open(outdir + previous_date + '.export.CSV', 'w') as outf:
                    if len(day_buffer) >= MIN_NUM_EVENTS:
                        for line in day_buffer[1:]:
                            outf.write("%s" % line)

                    print('wrote {}'.format(previous_date))

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
            filename = year + str(m).zfill(2) + '.zip'
            download_and_extract(filename)
    else:
        for m in range(0,13):
            filename = year + str(m).zfill(2) + '.zip'
            download_and_extract(filename)
            

elif year_i > 1978:
    filename = year + '.zip'
    download_and_extract(filename)




