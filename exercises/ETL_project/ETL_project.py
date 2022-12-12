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
import pandas as pd
import numpy as np
pre = df_pre.toPandas()
after = df_after.toPandas()

#3. Rerun the same ML training and scoring logic that you had created prior 
#   to this class, starting with the Pandas DataFrames you got in step 2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pre.merge(after, on='movie_title', how='inner')
df = df.drop(columns = ["num_critic_for_reviews","gross","movie_title","num_voted_users","num_user_for_reviews","movie_facebook_likes"])

df.drop_duplicates(inplace = True)

df = df.drop(columns = ["director_name","actor_1_name","actor_2_name","actor_3_name"])

df['content_rating'].fillna('R', inplace=True)
df['language'].fillna('English', inplace=True)

df["actor_1_facebook_likes"].fillna(653.813397, inplace=True)
df["actor_2_facebook_likes"].fillna(438.397510, inplace=True)
df["actor_3_facebook_likes"].fillna(299.271593, inplace=True)
df.dropna()

for i in ["UK","France","Germany","Spain","Italy","Norway","Denmark","Ireland","Netherlands","Romania","Sweden","Iceland","West Germany","Slovenia","Poland","Czech Republic","Bulgaria","Belgium"]:
    c = df["country"] != i
    df["country"].where(c, "Europe", inplace = True)
cat = ["USA","Europe"]
df['country'] = df.country.where(df.country.isin(cat), 'other')

l = df["language"] == "English"
df["language"].where(l, "Other", inplace = True)

x = df.drop(columns = ["imdb_score","content_rating"])
y = df["imdb_score"] 

x = pd.concat([x, x['genres'].str.get_dummies(sep='|')], axis=1)
x = x.drop("genres",axis = 1)

x_head_cat = ["language","country"]
x_head_num = x.columns[~x.columns.isin(x_head_cat)]

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    del dummies[dummies.columns[0]] 
    res = pd.concat([original_dataframe, dummies], axis=1)
    res.drop(feature_to_encode, axis=1, inplace=True) 
    return(res)
for categorical_feature in x_head_cat:
    x[categorical_feature] = x[categorical_feature].astype('category')
    x = encode_and_bind(x,categorical_feature)

seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)

rfreg = RandomForestRegressor(max_depth=10, min_samples_leaf =1, random_state=1).fit(x_train, y_train)

array_pred = np.round(rfreg.predict(x_val),0)
y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) 
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
print(val_pred)

val_pred = pd.concat([y_val,y_pred,x_val],axis=1)

from pyspark.sql.types import StringType, DateType, IntegerType, TimestampType, StructField, StructType

def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return DoubleType()
    elif f == 'float32': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)

df_pred = pandas_to_spark(val_pred)

#5. Write this DataFrame to the same S3 bucket dmacademy-course-assets under the prefix 
#   vlerick/<your_name>/ as JSON lines. It is likely Spark will create multiple files there. 
#   That is entirely normal and inherent to the distributed processing character of Spark.

# # Write the DataFrame to S3 as JSON lines
df_pred.write.json(f"s3a://{BUCKET}/vlerick/wannes/", mode="overwrite")