import pandas as pd
import seaborn as sns

PATH = "D:\!!!MAAykanat Dosyalar\Miuul\Python for Data Science\Görevler\A3-Gezinomi Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama\gezinomi_tanıtım"

df = pd.read_excel(PATH + "\gezinomi_tanıtım.xlsx")
print(df.head())