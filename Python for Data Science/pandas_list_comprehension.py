import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

############################################
# Assignment-1:
"""Capitalize the names of numeric variables in the car_crashes data by using the List Comprehension structure.
convert it to letter and add NUM at the beginning.
"""
###########################################
print("*"*100)

df_car_crush = sns.load_dataset("car_crashes")

print(df_car_crush.head())
print(df_car_crush.info())
print(df_car_crush.describe().T)

df_car_crush_columns=[("NUM_"+col).upper() for col in df_car_crush.columns]

print(df_car_crush_columns)

############################################
# Assignment-2:
"""Using the List Comprehension structure, the car_crashes data does not contain "no" in its name.
Write "FLAG" at the end of the variable names.
"""
###########################################
print("*"*100)

print([(col + "_FLAG").upper() if "NO" not in col.upper() else col.upper() for col in df_car_crush.columns ])

############################################
# Assignment-3:
"""Using the List Comprehension structure, 
you can find the variable names that are DIFFERENT from the ones given below.
Select the names of the variables and create a new dataframe.
og_list = ['abbrev', 'no_previoues']
"""
###########################################
print("*"*100)

og_list = ['abbrev', 'no_previous']

print([col for col in df_car_crush.columns if col not in og_list])

df_new = df_car_crush.loc[:, [col for col in df_car_crush.columns if col not in og_list]]

print(df_new.head())

############################################
# Assignment-4:
"""
Task 1: Identify the Titanic dataset from the Seaborn library.
Task 2: Find the number of male and female passengers in the Titanic data set.
Task 3: Find the number of unique values for each column.
Task 4: Find the number of unique values of the pclass variable.
Task 5: Find the number of unique values of the pclass and parch variables.
Task 6: Check the type of the embarked variable. Change its type to category and check again.
Task 7: Show all the information of those whose embarked value is C.
Task 8: Show all the information of those whose embarked value is not S.
Task 9: Show all the information about passengers who are women and under 30 years old.
Task 10: Show Fare the information of passengers higher than 500 or older than 70 years old.
Task 11: Find the sum of the null values in each variable.
Task 12: Remove the who variable from the dataframe.
Task 13: Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.
Task 14: Fill the empty values in the age variable with the median of the age variable.
Task 15: Find the sum, count, mean values of the variable survived in the breakdown of pclass and gender variables.
Task 16: Write a function that will give 1 to those under 30 and 0 to those equal to or above 30. titanic data using the function you wrote
Create a variable named age_flag in the set. (use apply and lambda structures)
Task 17: Define the Tips dataset within the Seaborn library.
Task 18: Find the sum, min, max and average of total_bill values according to the categories (Dinner, Lunch) of the Time variable.
Task 19: Find the sum, min, max and average of total_bill values according to days and time.
Task 20: Find the sum, min, max and average of total_bill and type values of lunch time and female customers according to day.
Task 21: What is the average of orders with size less than 3 and total_bill greater than 10? (use loc)
Task 22: Create a new variable named total_bill_tip_sum. Let it give the total bill and tip paid by each customer.
Task 23: Sort from largest to smallest according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe.
"""
###########################################
print("*"*100)

#Task 1:
df_titanic = sns.load_dataset("titanic")
print(df_titanic.head())

#Task 2:
print(df_titanic["sex"].value_counts())

#Task 3:
print(df_titanic.nunique())

#Task 4:
print("# of Unique variable of pclass: ",df_titanic["pclass"].nunique())

#Task 5:
print("# of Unique variable of pclass and parch:\n",df_titanic[["pclass","parch"]].nunique())

#Task 6:
print("1-Type of embarked variable: ",df_titanic["embarked"].dtype)
df_titanic["embarked"] = df_titanic["embarked"].astype("category")
print("2-Type of embarked variable: ",df_titanic["embarked"].dtype)

#Task 7:
print(df_titanic[df_titanic["embarked"] == "C"])

#Task 8:
print(df_titanic[df_titanic["embarked"] != "S"])

#Task 9:
print(df_titanic[(df_titanic["sex"]=="female") & (df_titanic["age"]<30)].head())

#Task 10:
print(df_titanic[(df_titanic["fare"]>500) | (df_titanic["age"]>70)].head())

#Task 11:
print(df_titanic.isnull().sum())

#Task 12:
df_titanic.drop("who", axis=1, inplace=True)
print(df_titanic.head())

#Task 13:
print(type(df_titanic["deck"].mode()))
print("Mode of deck variable: ",df_titanic["deck"].mode()[0])
print(df_titanic["deck"].isnull().sum())
print(df_titanic["deck"].fillna(df_titanic["deck"].mode()[0], inplace=True))
print(df_titanic["deck"].isnull().sum())

#Task 14:
print(df_titanic["age"].isnull().sum())
print(df_titanic["age"].fillna(df_titanic["age"].median(), inplace=True))
print(df_titanic["age"].isnull().sum())

#Task 15:
print(df_titanic.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]}))

#Task 16:
def age_flag(x):
    return 1 if x < 30 else 0

df_titanic["age_flag"] = df_titanic["age"].apply(lambda x: age_flag(x))
print(df_titanic.head())

#Task 17:
df_tips = sns.load_dataset("tips")
print(df_tips.head())

#Task 18:
print(df_tips.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]}))

#Task 19:
print(df_tips.groupby(["day","time"]).agg({"total_bill": ["sum", "min", "max", "mean"]}))

#Task 20:
print(df_tips[(df_tips["time"]=="Lunch") & (df_tips["sex"]== "Female")].head(50).sort_values(by=["day"], ascending=False))
print(df_tips[(df_tips["time"]=="Lunch") & (df_tips["sex"]== "Female")].groupby(["day"]).agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                                              "tip": ["sum", "min", "max", "mean"]}))

#Task 21:
print(df_tips.loc[(df_tips["size"]<3) & (df_tips["total_bill"]>10)].mean())

#Task 22:
df_tips["total_bill_tip_sum"] = df_tips["total_bill"] + df_tips["tip"]
print(df_tips.head())

#Task 23:
df_new=df_tips.sort_values(by=["total_bill_tip_sum"], ascending=False)[:30]
print(df_new)