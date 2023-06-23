# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import TimestampType
from functools import reduce
from typing import List
from operator import add
import datetime

# COMMAND ----------

data_dict = {
    "aapl": 0.46,
    "bac": 0.1,
    "cvx": 0.09,
    "ko": 0.085,
    "axp": 0.08,
    "oxy": 0.05,
    "khc": 0.05,
    "mco": 0.035,
    "atvi": 0.02,
    "byddf": 0.02,
    "hpq": 0.01
}


# COMMAND ----------

calculate_returns = lambda previous_price, current_price: (current_price - previous_price) / previous_price

# COMMAND ----------

def company_liquidity(tickr):
    df = spark.sql(f"SELECT * FROM `hive_metastore`.`default`.`{tickr}_daily_ret`") \
        .withColumnRenamed("Adj Close", "CurrentDayPrice") \
        .withColumn("PreviousDayPrice", lag("CurrentDayPrice").over(Window.orderBy("Date"))) \
        .na.drop("any") \
        .withColumn("DailyReturns", abs(calculate_returns(col("PreviousDayPrice"), col("CurrentDayPrice")))) \
        .withColumn("DollarTradingVolume", col("CurrentDayPrice") * col("Volume")) \
        .withColumn("LIQ", col("DailyReturns") / col("DollarTradingVolume")) \
        .withColumn("Date", date_format(col("Date"), "yyyy-MM")) \
        .groupBy("Date").agg(avg(col("LIQ")).alias(f"{tickr}")) \
    
    return df

# COMMAND ----------

# Create a list of DataFrames by iterating over the dictionary elements
dfs_liq = [company_liquidity(ticker) for ticker, weight in data_dict.items()]

# Use reduce operation to join all DataFrames in the list
stock_liquidity_df = reduce(lambda df1, df2: df1.join(df2, "Date"), dfs_liq)
# The lambda function takes two DataFrame arguments (df1 and df2) and joins them on the "Date" column using the join method. The resulting DataFrame is returned as the output of the lambda function.
# The reduce function is used to iteratively join DataFrames from the dfs list based on a common column "Date". 

display(stock_liquidity_df)

# COMMAND ----------

def sum_weighted_products(df, mapper, name):

    tickrList = [f"{key}" for key, value in mapper.items()] 

    for col_name, factor in mapper.items():
        df = df.withColumn(col_name, col(col_name) * factor)

    resultDf = df.withColumn(name ,reduce(add, [col(x) for x in tickrList])).select("Date", name)

    return resultDf

# COMMAND ----------

portfolio_liquidity_df = sum_weighted_products(stock_liquidity_df, data_dict, "LIQ")
display(portfolio_liquidity_df)

# COMMAND ----------

def companyMonthlyReturns(tickr):
    df = spark.sql(f"SELECT * FROM `hive_metastore`.`default`.`{tickr}`") \
        .select(col("Date"), col("Adj Close").alias("CurrentMonthPrice")) \
        .withColumn("PreviousMonthPrice", lag("CurrentMonthPrice").over(Window.orderBy("Date"))) \
        .na.drop("any") \
        .withColumn(f"{tickr}", calculate_returns(col("PreviousMonthPrice"), col("CurrentMonthPrice"))) \
        .select("Date", f"{tickr}") \
        .filter(year(col("Date")) > 2017) \
        .filter(year(col("Date")) < 2023) \
        .withColumn("Date", date_format(col("Date"), "yyyy-MM")) \
    
    return df

# COMMAND ----------

# Create a list of DataFrames by iterating over the dictionary elements
dfs_ret = [companyMonthlyReturns(ticker) for ticker, weight in data_dict.items()]

# Use reduce operation to join all DataFrames in the list
stock_returns_df = reduce(lambda df1, df2: df1.join(df2, "Date"), dfs_ret)
# The lambda function takes two DataFrame arguments (df1 and df2) and joins them on the "Date" column using the join method. The resulting DataFrame is returned as the output of the lambda function.
# The reduce function is used to iteratively join DataFrames from the dfs list based on a common column "Date". 

display(stock_returns_df)

# COMMAND ----------

portfolio_returns_df = sum_weighted_products(stock_returns_df, data_dict, "portfolio_return")
display(portfolio_returns_df)

# COMMAND ----------

ff_data = spark.sql("SELECT * FROM `hive_metastore`.`default`.`ff_5`") \
    .withColumn("Date", from_unixtime(unix_timestamp(col("Date").cast("string"), "yyyyMM"), "yyyy-MM")) \
    .filter(year(col("Date")) > "2017") \
    .filter(year(col("Date")) < "2023") \
    .withColumn("Mkt-RF", col("Mkt-RF") / 100) \
    .withColumn("SMB", col("SMB") / 100) \
    .withColumn("HML", col("HML") / 100) \
    .withColumn("RF", col("RF") / 100)

display(ff_data)

# COMMAND ----------

portfolio_df = ff_data.join(portfolio_returns_df, ["Date"]).join(portfolio_liquidity_df, ["Date"]) \
.withColumn("portfolio_excess_return", col("portfolio_return")-col("RF"))
display(portfolio_df)

# COMMAND ----------

def regression_model(feature_list):
    assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    labeledData = assembler.transform(portfolio_df).select("portfolio_excess_return", "features")

    lr = LinearRegression(labelCol="portfolio_excess_return", featuresCol="features")
    model = lr.fit(labeledData)

    beta = model.coefficients[0]
    alpha = model.intercept

    print("alpha =", alpha)
    print("beta =", beta)

    predictions = model.transform(labeledData)
    evaluator = RegressionEvaluator(labelCol="portfolio_excess_return", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print("R-squared:", r2)

    evaluator = RegressionEvaluator(labelCol="portfolio_excess_return", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("RMSE:", rmse)

# COMMAND ----------

#CAPM Model
regression_model(["Mkt-RF"])

# COMMAND ----------

#FF- 3 Model
regression_model(["Mkt-RF", "SMB", "HML"])

# COMMAND ----------

#FF- 3 + LIQ Model
regression_model(["Mkt-RF", "SMB", "HML", "LIQ"])

# COMMAND ----------

#FF- 5 Model
regression_model(["Mkt-RF", "SMB", "HML", "RMW", "CMA"])
