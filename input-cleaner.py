#!/usr/bin/env python

'''
Script which does some initial cleansing of the data.

First argument should be the input directory, which should have files in the
form YYYYMMDD.export.CSV, with one file per day expected to be there.

Next argument should be the output directory, which is created on demand.

The output directory is populated with cleansed TSV files.

E.g.:

/fs/in

20000101.export.CSV
20000102.export.CSV
20001213.export.CSV

Then input-cleaner.py /fs/in /fs/out will end up with:

/fs/out

20000101.tsv
20000102.tsv
20001213.tsv
'''



