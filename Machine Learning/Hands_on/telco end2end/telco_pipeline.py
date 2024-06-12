#######################################################
# End-to-End Telco Churn Machine Learning Pipeline II #
#######################################################

def telco_data_prep(dataframe):
    target=["CHURN"]

    dataframe.columns = [col.upper() for col in dataframe.columns]
    
    dataframe["TOTALCHARGES"] = pd.to_numeric(dataframe["TOTALCHARGES"], errors="coerce")
    dataframe["CHURN"] = dataframe["CHURN"].apply(lambda x: 1 if x =="Yes" else 0)

    dataframe.drop("CUSTOMERID", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grap_column_names(dataframe)

    for col in num_cols:
        replace_with_thresholds(dataframe, col)
        dataframe[col].fillna(dataframe[col].median(), inplace=True)
    
    ########################
    ### FEATURE CREATION ###
    ########################
    # Creating an annual categorical variable from a tenure variable
    dataframe.loc[(dataframe["TENURE"]>=0) & (dataframe["TENURE"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
    dataframe.loc[(dataframe["TENURE"]>12) & (dataframe["TENURE"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
    dataframe.loc[(dataframe["TENURE"]>24) & (dataframe["TENURE"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
    dataframe.loc[(dataframe["TENURE"]>36) & (dataframe["TENURE"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
    dataframe.loc[(dataframe["TENURE"]>48) & (dataframe["TENURE"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
    dataframe.loc[(dataframe["TENURE"]>60) & (dataframe["TENURE"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

    # Specify customers with 1 or 2 years of contract as Engaged
    dataframe["NEW_ENGAGED"] = dataframe["CONTRACT"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

    # People who do not receive any support, backup or protection
    dataframe["NEW_NOPROT"] = dataframe.apply(lambda x: 1 if (x["ONLINEBACKUP"] != "Yes") or (x["DEVICEPROTECTION"] != "Yes") or (x["TECHSUPPORT"] != "Yes") else 0, axis=1)

    # Customers who have monthly contracts and are young
    dataframe["NEW_YOUNG_NOT_ENGAGED"] = dataframe.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and (x["SENIORCITIZEN"] == 0) else 0, axis=1)

    # Total number of services received by the person
    dataframe['NEW_TOTALSERVICES'] = (dataframe[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY',
                                        'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT',
                                        'STREAMINGTV', 'STREAMINGMOVIES']]== 'Yes').sum(axis=1)

    # People who receive any streaming service
    dataframe["NEW_FLAG_ANY_STREAMING"] = dataframe.apply(lambda x: 1 if (x["STREAMINGTV"] == "Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

    # Does the person make automatic payments?
    dataframe["NEW_FLAG_AUTOPAYMENT"] = dataframe["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

    # Average monthly payment
    dataframe["NEW_AVG_CHARGES"] = dataframe["TOTALCHARGES"] / (dataframe["TENURE"] + 1)

    # Rate of change the current price compared to the average price
    dataframe["NEW_INCREASE"] = dataframe["NEW_AVG_CHARGES"] / dataframe["MONTHLYCHARGES"]

    # Fee per service
    dataframe["NEW_AVG_SERVICE_FEE"] = dataframe["MONTHLYCHARGES"] / (dataframe['NEW_TOTALSERVICES'] + 1)

    cat_cols, num_cols, cat_but_car = grap_column_names(dataframe)

    ###ENCODING###
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == "O"]

    for col in binary_cols:
        df = label_encoder(df, col)
    
    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in target]
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    ###STANDARDIZATION###
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
