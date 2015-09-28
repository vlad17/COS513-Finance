import urllib
from os import getcwd
import sys


gdelt_site = 'http://data.gdeltproject.org/events/'
url_opener = urllib.URLopener()

year = sys.argv[1]



def download_file(filename):
    filepath = gdelt_site + filename

    try:
        url_opener.retrieve(filepath, filename)
        print 'downloaded ' + filename
    except:
        print 'no file ' + filepath



if int(year) > 2005: # year = 2006 or later
    for month in range(1, 13):

        if int(year) > 2013: # year = 2013 or later
            for day in range(1, 32):
                filename = year + str(month).zfill(2) + str(day).zfill(2) + '.export.CSV.zip'
                download_file(filename)

        elif int(year) == 2013:
            if month < 4:
                filename = year + str(month).zfill(2) + '.zip'
            else:
                filename = year + str(month).zfill(2) + str(day).zfill(2) + '.export.CSV.zip'
            download_file(filename)

        else:
            filename = year + str(month).zfill(2) + '.zip'
            download_file(filename)

else: # year = 2005 or earlier
    filename = year + '.zip'
    download_file(filename)
