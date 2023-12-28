import dataset_handle as dh

df_titanic = dh.load_dataset("titanic.csv")

dh.dataset_details(df_titanic)