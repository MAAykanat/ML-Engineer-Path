import dataset_handle as dh
import seaborn as sns
import matplotlib.pyplot as plt

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    
    interquantile_range = quartile3 - quartile1
    
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)
    
    if dataframe[dataframe[col_name] <lower_limit] | dataframe[dataframe[col_name] > upper_limit]:
        return True
    else:
        return False
df_titanic = dh.load_dataset("titanic.csv")
dh.dataset_details(df_titanic)
dh.plot_hist(df_titanic, "Fare")
dh.plot_boxplot(df_titanic, "Fare")

#############################################
# Outlier Detection
#############################################
# Outlier Detection with Graphs
# Boxplot
#############################################
# Boxplot is a graphical method to visualize the dhstribution of data based on 
# the five-number summary: minimum, first quartile, medhan, third quartile, and maximum.
#############################################

# sns.boxplot(x=df_titanic["Fare"])
# plt.show()

low, up = outlier_thresholds(df_titanic, "Fare")
print("Low limit: ", low)
print("Up limit: ", up)

print(df_titanic[(df_titanic["Age"] < low)])


# print(df_titanic[(df_titanic["Fare"] > up) | (df_titanic["Fare"] < low)])