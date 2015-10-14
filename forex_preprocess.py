import csv

_COUNTRIES_OF_INTEREST = ["USA", "GBR"]

# List of all the column names
_COLUMNS = ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name",
            "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code",
            "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name",
            "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code",
            "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode",
            "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
            "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code",
            "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
            "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
            "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat",
            "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED"]

# Create a mapping from column name to its indices
_INDICES = {}
for i in xrange(len(_COLUMNS)):
    _INDICES[_COLUMNS[i]] = i

_EMTPY_CAMEO = "empty"


def load_csv(filename):
    """
    :return: list of rows (where each row is a list)
    """
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        return list(reader)


def keep_row(row):
    """
    :param row: a list for the row in the data
    :return: True if we should keep row; False if we should discard row
    """
    if row[_INDICES["Actor1CountryCode"]] in _COUNTRIES_OF_INTEREST or \
        row[_INDICES["Actor2CountryCode"]] in _COUNTRIES_OF_INTEREST:
        return True

    return False

def new_column_names(country, cameo):
    """
    Given a country code and the cameo code, return the names of the new
    columns for positive and negative impact
    :param country: string
    :param cameo: string
    :return: 2-tuple of strings
    """
    if cameo == "":
        cameo = _EMTPY_CAMEO
    positive = "%s_%s_pos" % (country, cameo)
    negative = "%s_%s_neg" % (country, cameo)
    return (positive, negative)


def preprocess(raw_data):

    # TODO: add filtering by country here

    filtered_data = []
    for row in raw_data:
        if keep_row(row):
            filtered_data.append(row)


    # Get a list of unique country codes
    # country_code_1_column = map(lambda row: row[_INDICES["Actor1CountryCode"]], filtered_data)
    #  country_code_2_column = map(lambda row: row[_INDICES["Actor2CountryCode"]], filtered_data)
    # unique_country_codes = set(country_code_1_column).union(set(country_code_2_column))
    # Note that in this case, we just want USA and GBR
    unique_country_codes = set(_COUNTRIES_OF_INTEREST)

    # Get a list of unique cameo types
    actor_1_type_1_column = map(lambda row: row[_INDICES["Actor1Type1Code"]], filtered_data)
    actor_1_type_2_column = map(lambda row: row[_INDICES["Actor1Type2Code"]], filtered_data)
    actor_1_type_3_column = map(lambda row: row[_INDICES["Actor1Type3Code"]], filtered_data)
    actor_2_type_1_column = map(lambda row: row[_INDICES["Actor2Type1Code"]], filtered_data)
    actor_2_type_2_column = map(lambda row: row[_INDICES["Actor2Type2Code"]], filtered_data)
    actor_2_type_3_column = map(lambda row: row[_INDICES["Actor2Type3Code"]], filtered_data)

    unique_cameo_types = set(actor_1_type_1_column).union(set(actor_1_type_2_column)).union(set(actor_1_type_3_column)
        ).union(set(actor_2_type_1_column)).union(set(actor_2_type_2_column)).union(set(actor_2_type_3_column))

    # The *2 is for both positive impact and negative impact
    # The +1 is for the date column
    new_columns = [""] * (len(unique_country_codes) * len(unique_cameo_types) * 2 + 1)
    new_indices = {"SQLDATE": 0}
    new_columns[0] = "SQLDATE"
    idx = 1
    for cameo in unique_cameo_types:
        for country in unique_country_codes:
            (pos, neg) = new_column_names(country, cameo)
            new_columns[idx] = pos
            new_columns[idx + 1] = neg
            new_indices[pos] = idx
            new_indices[neg] = idx + 1
            idx += 2
    # keep running averages of numArticles etc in a dictionary indexed by date
    running_average_data = {}
    current_day = None
    count = [0,0,0,0] #NumArticles, NumMentions, NumSources, number of articles in a day
    for row in filtered_data:
        # aggregate into individual days
        if row[_INDICES["SQLDATE"]] != current_day:
            current_day = row[_INDICES["SQLDATE"]]
            running_average_data[current_day] = count 
            count = [int(row[_INDICES["NumArticles"]]),
                    int(row[_INDICES["NumMentions"]]),
                    int(row[_INDICES["NumSources"]]),
                    1]
        else:
            count[0]+= int(row[_INDICES["NumArticles"]])   
            count[1]+= int(row[_INDICES["NumMentions"]])
            count[2]+= int(row[_INDICES["NumSources"]])
            count[3]+= 1
    for day in running_average_data.keys():
        running_average_data[day] = [day[1]/day[3], day[2]/day[3], day[3]/day[3], day[3]]

    processed_data = []
    # The first row to write are the headers
    new_row = new_columns 
    # The current day to process
    current_day = None
    for row in filtered_data:
        # aggregate into individual days
        if row[_INDICES["SQLDATE"]] != current_day:
            processed_data.append(new_row)
            current_day = row[_INDICES["SQLDATE"]]
            new_row = [0.0] * len(new_columns)
            new_row[new_indices["SQLDATE"]] = current_day

        # The heuristics for calculating positive and negative impact are here

        # normalize by the day
        norm_NumArticles = int(row[_INDICES["NumArticles"]])/running_average_data[current_day][0]
        norm_NumArticles = int(row[_INDICES["NumMentions"]])/running_average_data[current_day][1]
        norm_NumArticles = int(row[_INDICES["NumSources"]])/running_average_data[current_day][2]


        # Heuristic 1: NumArticles * Goldstein scale
        h1 = norm_NumArticles * float(row[_INDICES["GoldsteinScale"]])

        # Heuristic 2: NumArticles * AvgTone scale
        h2 = norm_NumArticles * float(row[_INDICES["AvgTone"]])

        def quadclass_impact(quadclass):
            """
            Given a quad class, return a measure of its impact
            cooperation maps to {1, 2}
            conflict maps to {-1, -2}
            """
            quadclass = int(quadclass)
            if quadclass in [3, 4]:
                return -(quadclass - 2)
            return quadclass

        # Heuristic 3: NumArticles * QuadClass scale
        h3 = norm_NumArticles * float(quadclass_impact(row[_INDICES["QuadClass"]]))

        h = h3

        country = row[_INDICES["Actor1CountryCode"]]
        # Figure out how many CAMEO codes for the first actor there are
        if (country in _COUNTRIES_OF_INTEREST):
            actor_1_cameo = set([row[_INDICES["Actor1Type1Code"]], row[_INDICES["Actor1Type2Code"]],
                                 row[_INDICES["Actor1Type3Code"]]])
            actor_1_cameo.discard("")
            if not actor_1_cameo:
                # There is no CAMEO code
                actor_1_cameo = [""]
            for cameo in actor_1_cameo:
                new_cn = new_column_names(country, cameo)
                if h > 0:
                    new_row[new_indices[new_cn[0]]] += h / len(actor_1_cameo)
                else:
                    new_row[new_indices[new_cn[1]]] += h / len(actor_1_cameo)


        country = row[_INDICES["Actor2CountryCode"]]
        # Figure out how many CAMEO codes for the second actor there are
        if (country in _COUNTRIES_OF_INTEREST):
            actor_2_cameo = set([row[_INDICES["Actor2Type1Code"]], row[_INDICES["Actor2Type2Code"]],
                                 row[_INDICES["Actor2Type3Code"]]])
            actor_2_cameo.discard("")
            if not actor_2_cameo:
                # There is no CAMEO code
                actor_2_cameo = [""]
            for cameo in actor_2_cameo:
                new_cn = new_column_names(country, cameo)
                if h > 0:
                    new_row[new_indices[new_cn[0]]] += h / len(actor_2_cameo)
                else:
                    new_row[new_indices[new_cn[1]]] += h / len(actor_2_cameo)

    processed_data.append(new_row)

    return processed_data


def write_csv(filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for row in processed_data:
            writer.writerow(row)


if __name__ == '__main__':
    input_file = '/Users/tomwu/Google Drive/COS513 Project Folder/gdelt_2005.csv'
    input_file = 'gdelt_filtered_2005.csv'
    raw_data = load_csv(input_file)
    processed_data = preprocess(raw_data)
    write_csv('preprocessed_data_2005.csv')
    # write_csv('gdelt_filtered_2005.csv')


    print "Finished running"
