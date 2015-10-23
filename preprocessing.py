"""
Usage: python preprocessing.py inputfile outputfile

Accepts tab-separated GDELT record files of NUM_FIELDS (or NUM_FIELDS - 1)
columns (URL, column , and ouptuts a cleaned, preprocessed version to outputfile.

Assumes basename for input file is of the form YYYYMMDD.export.CSV.

NOTES:
* CATEGORICAL: encoded as integers from [0, n-1) for n-1 distinct values.
This is done for space concerns, and these columns should be expanded with 
sklearn.preprocessing.OneHotEncoder after. They use 0 for missing values.
* STRING: They should be expanded with Word2Vec before processing as well.
These are already pretty clean (no punctuation, etc.). They are all lowercase.
* NUMERIC: A 4-byte floating point value.
* The output column EventDaysSinceEpoch is probably not as useful for prediction
as (Today - DaysSinceEpoch), but this value would change day-to-day. A
transformation before processing may be in order. Same goes for
PublishedDaysSinceEpoch.
* NO whitening or normalization is performed, and no scales or transformations
  are applied.
* CAMEOCode* interactions are nonlinear - consider adding features with
  different polynomial values of their categories (e.g., Code1 * Code2 for each
  one-hot term).

Output Schema:
***** Importance-related *****
1. EventDaysSinceEpoch - Days since Unix epoch that event occured. NUMERIC.
2. PublishedDaysSinceEpoch - Days since Unix epoch that article was published. 
   NUMERIC. 
3. IsVerbal - 0 or 1. NUMERIC.
4. GoldsteinScale - -10 to 10 scale of "weight" of event. NUMERIC.
5. NumMentions - integer number of mentions of event in article. NUMERIC.
6. NumSources - number of primary source mentions. NUMERIC.
7. NumArticles - number of secondary source mentions. NUMERIC.
8. AvgTone - -100 to 100 indication of positivity/extremity of tone. NUMERIC.
***** Topic-related *****
9. CAMEOCode1 - Describes high-level event attributes. 20. CATEGORICAL.
10. CAMEOCode2 - Mid-level event attributes. 100. CATEGORICAL.
11. CAMEOCode3 - Low-level event attributes. 10. CATEGORICAL.
12. IsCooperative - 0 or 1. NUMERIC.
13. Actor1Country - Country "index". CATEGORICAL.
14. Actor2Country - Country "index". CATEGORICAL.
15. Actor1Geo_Type - CATEGORICAL
16. Actor2Geo_Type - CATEGORICAL
17. ActionGeo_Type - CATEGORICAL
18. ActionGeo_Lat - NUMERIC
19. ActionGeo_Long - NUMERIC
20. Actor1Name - STRING.
21. Actor2Name - STRING.
TODO: We can add more stuff, but what to do if it's empty? Right now, we drop
      the row for important (non-name) empty values. What if we impute?
      We can add (Actor*|Action)Geo_CountryCode CATEGORICAL.
      Also we can add Actor(1|2)Geo_(Lat|Long).

Input Schema (fields without description are discarded)
1. GlobalEventID
2. SQLDATE - YYYYMMDD date
3-6. MonthYear Year FractionDate Actor1Code
7. Actor1Name - Actor 1 name
8. Actor1CountryCode - ISO 3166 Alpha-3 Country Code
9-16. Actor1KnownGroupCode Actor1EthnicCode Actor1Religion1Code
      Actor1Religion2Code Actor1Type1Code Actor1Type2Code Actor1Type3Code
      Actor2Code
17. Actor2Name - Actor 2 name
18. Actor2CountryCode - ISO 3166 Alpha-3 Country Code
19-25. Actor2KnownGroupCode Actor2EthnicCode Actor2Religion1Code
       Actor2Religion2Code, Actor2Type1Code, Actor2Type2Code, Actor2Type3Code
26. IsRootEvent - Whether this was the "cause" event
27. EventCode - Categorical description of event
28. EventBaseCode - Prefix of EventCode, delimited at a "significance" level
29. EventRootCode - Prefix of EventBaseCode
30. QuadClass - Description of event sentiment as 4 categories
31. GoldsteinScale - Numeric scale of weight of event
32. NumMentions - integer number of mentions of event in article
33. NumSources - number of primary source mentions
34. NumArticles - number of secondary source mentions
35. AvgTone - -100 to 100 indication of positivity/extremity of tone.
36. Actor1Geo_Type - Scope of geographic region affected.
37-39. Actor1Geo_FullName Actor1Geo_CountryCode Actor1Geo_ADM1Code
40. Actor1Geo_Lat - lattitude
41. Actor1Geo_Long - longitude
42. Actor1Geo_FeatureID
43. Actor2Geo_Type - see above
44-46 Actor2Geo_FullName Actor2Geo_CountryCode Actor2Geo_ADM1Code
47. Actor2Geo_Lat - see above
48. Actor2Geo_Long - see above
49. Actor2Geo_FeatureID
50. ActionGeo_Type - see above
51-53. ActionGeo_FullName ActionGeo_CountryCode ActionGeo_ADM1Code
54. ActionGeo_Lat - see above
55. ActionGeo_Long - see above
56-58. ActionGeo_FeatureID DATEADDED [URL]
"""

import csv
from datetime import datetime
import sys
import os
import re
from iso3166 import countries_by_alpha3

column_idx = {
    'GlobalEventID' : 0, 'SQLDATE' : 1, 'MonthYear' : 2, 'Year' : 3,
    'FractionDate' : 4, 'Actor1Code' : 5, 'Actor1Name' : 6,
    'Actor1CountryCode' : 7, 'Actor1KnownGroupCode' : 8, 'Actor1EthnicCode' : 9,
    'Actor1Religion1Code' : 10, 'Actor1Religion2Code' : 11,
    'Actor1Type1Code' : 12, 'Actor1Type2Code' : 13, 'Actor1Type3Code' : 14,
    'Actor2Code' : 15, 'Actor2Name' : 16, 'Actor2CountryCode' : 17,
    'Actor2KnownGroupCode' : 18, 'Actor2EthnicCode' : 19, 
    'Actor2Religion1Code' : 20, 'Actor2Religion2Code' : 21,
    'Actor2Type1Code' : 22, 'Actor2Type2Code' : 23, 'Actor2Type3Code' : 24,
    'IsRootEvent' : 25, 'EventCode' : 26, 'EventBaseCode' : 27, 
    'EventRootCode' : 28, 'QuadClass' : 29, 'GoldsteinScale' : 30,
    'NumMentions' : 31, 'NumSources' : 32, 'NumArticles' : 33,
    'AvgTone' : 34, 'Actor1Geo_Type' : 35, 'Actor1Geo_FullName' : 36,
    'Actor1Geo_CountryCode' : 37, 'Actor1Geo_ADM1Code' : 38,
    'Actor1Geo_Lat' : 39, 'Actor1Geo_Long' : 40, 'Actor1Geo_FeatureID' : 41,
    'Actor2Geo_Type' : 42, 'Actor2Geo_FullName' : 43,
    'Actor2Geo_CountryCode' : 44, 'Actor2Geo_ADM1Code' : 45,
    'Actor2Geo_Lat' : 46, 'Actor2Geo_Long' : 47,
    'Actor2Geo_FeatureID' : 48, 'ActionGeo_Type' : 49,
    'ActionGeo_FullName' : 50, 'ActionGeo_CountryCode' : 51,
    'ActionGeo_ADM1Code' : 52, 'ActionGeo_Lat' : 53,
    'ActionGeo_Long' : 54, 'ActionGeo_FeatureID' : 55,
    'DATEADDED' : 56, 'URL' : 57}

NUM_FIELDS = 58

alpha3s = countries_by_alpha3.keys()
alpha3s = dict(zip(alpha3s, range(1, len(alpha3s))))

def is_verbal(quadclass):
    return quadclass == 1 or quadclass == 3

def is_cooperation(quadclass):
    return quadclass == 1 or quadclass == 4

def bool_int(boolean):
    return 1 if boolean else 0

def parse_day(day):
    unix_epoch = datetime(1970, 1, 1)
    parsed = datetime(int(day[:4]), int(day[4:6]), int(day[6:8]))
    return (parsed - unix_epoch).days

def zerostr(s):
    return s if s else "0"

def clean_row(row, day):
    """ Performs the output schema -> input schema transformation described
        above. """
    
    new_row = []

    ## Importance-related columns

    # EventDaysSinceEpoch
    new_row.append(row[column_idx['SQLDATE']])
    if not new_row[-1]: return None

    # PublishedDaysSinceEpoch
    new_row.append(day)

    # IsVerbal
    quadclass = row[column_idx['QuadClass']]
    if not quadclass: return None
    quadclass = int(quadclass)
    new_row.append(bool_int(is_verbal(quadclass)))

    for col in ['GoldsteinScale', 'NumMentions', 'NumSources', 
                   'NumArticles', 'AvgTone']:
        new_row.append(row[column_idx[col]])
        if not new_row[-1]: return None

    ## Topic-related columns
    
    # CAMEOCode1
    level1 = row[column_idx['EventRootCode']]
    new_row.append(zerostr(level1))

    # CAMEOCode2
    level12 = row[column_idx['EventBaseCode']]
    new_row.append(zerostr(level12[len(level1):]))

    # CAMEOCode3
    level123 = row[column_idx['EventCode']]
    new_row.append(zerostr(level123[len(level12):]))

    # IsCooperative
    new_row.append(bool_int(is_cooperation(quadclass)))

    # Actor1Country
    actor1cc = row[column_idx['Actor1CountryCode']]
    actor1cc = alpha3s[actor1cc] if actor1cc in alpha3s else 0
    new_row.append(actor1cc)

    # Actor2Country
    actor2cc = row[column_idx['Actor2CountryCode']]
    actor2cc = alpha3s[actor2cc] if actor2cc in alpha3s else 0
    new_row.append(actor2cc)

    for col in ['Actor1Geo_Type', 'Actor2Geo_Type', 'ActionGeo_Type']:
        new_row.append(zerostr(row[column_idx[col]]))

    for col in ['ActionGeo_Lat', 'ActionGeo_Long']:
        new_row.append(row[column_idx[col]])
        if not new_row[-1]: return None

    for col in ['Actor1Name', 'Actor2Name']:
        new_row.append(row[column_idx[col]])

    return new_row

compatibleFile = re.compile('\d\d\d\d\d\d\d\d\.export\.CSV')

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    inbase = os.path.basename(inputfile)
    if not compatibleFile.match(inbase):
        print("inputfile should match YYYYMMDD.export.CSV")
        return 1

    news_day = parse_day(inbase[:8])
    
    print("Converting", inputfile, "to", outputfile)
    
    tot_rows = 0
    dropped_rows = 0
    with open(inputfile, 'r') as in_csv, open(outputfile, 'w') as out_csv:
        reader = csv.reader(in_csv, delimiter='\t')
        writer = csv.writer(out_csv, delimiter='\t')
        for row in reader:
            tot_rows += 1
            cleaned_row = clean_row(row, news_day)
            if cleaned_row:
                writer.writerow(cleaned_row)
            else:
                dropped_rows += 1
    print("Dropped", dropped_rows, "of", tot_rows)

    return 0

if __name__ == "__main__":
    main()

















