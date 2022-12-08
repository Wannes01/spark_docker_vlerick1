from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/after_release.csv"
KEY2 = "vlerick/pre_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

#1. Read the CSV data from this S3 bucket using PySpark
df_after = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
df_pre = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

df_pre.show()
df_after.show()

#2. Convert the Spark DataFrames to Pandas DataFrames
pre = df_pre.toPandas()
after = df_after.toPandas()

#3. Rerun the same ML training and scoring logic that you had created prior 
#   to this class, starting with the Pandas DataFrames you got in step 2

