import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
import findspark

findspark.init()

spark = SparkSession.builder.appName("regression").getOrCreate()

df = spark.read.csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv", header=True, inferSchema=True)

print(df.printSchema())
