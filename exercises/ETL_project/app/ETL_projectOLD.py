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

# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

## i. Data preparation

### Deal with missing values
pre = pre[pre.isnull().sum(axis=1) < 5]

#### Impute categorical values in pre
catPre=['content_rating','language','country']
cat_imputer = SimpleImputer(strategy = "most_frequent")
pre[catPre] = cat_imputer.fit_transform(pre[catPre])

#### Impute numerical values in pre
numPre = pre.select_dtypes(include = 'float64').columns.tolist()
num_imputer = SimpleImputer(strategy='median')
pre[numPre] = num_imputer.fit_transform(pre[numPre])

#### Impute numerical values in after
numAfter = after.select_dtypes(include = 'float64').columns.tolist()
num_imputer = SimpleImputer(strategy='median')
after[numAfter] = num_imputer.fit_transform(after[numAfter])

### Removing columns with high correlation
pre.drop(['actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes'], inplace=True, axis=1)

### Group catergories with low frequency
#### Language
pre.drop(['language'], inplace=True, axis=1)

#### Country
value_counts_country=pre['country'].value_counts()
mostFreqCountries = value_counts_country[:2].index
pre['country'] = pre.country.where(pre.country.isin(mostFreqCountries),'Other')

#### Content rating
value_counts_content_rating=pre['content_rating'].value_counts()
mostFreqContent_rating = value_counts_content_rating [:3].index
pre['content_rating'] = pre.content_rating.where(pre.content_rating.isin(mostFreqContent_rating),'Other')

### Remove duplicates
pre.drop_duplicates(inplace = True)
after.drop_duplicates(inplace = True)

### Split genre into seperate columns
pre = pd.concat([pre, pre['genres'].str.get_dummies()], axis=1)
pre.drop(['genres','Western'], inplace=True, axis=1)

### Merging pre and after-release information
merged=pd.merge(pre, after,how='inner', on='movie_title')

### Removing unneccesary columns
merged.drop(['director_name','actor_1_name','actor_2_name','actor_3_name','movie_title','num_critic_for_reviews','gross','num_voted_users','num_user_for_reviews','movie_facebook_likes'], inplace=True, axis=1)

### Preprocessing: Data transformations
#Change the datatype of int to float, otherwise standardization will not work
merged['cast_total_facebook_likes'] = merged['cast_total_facebook_likes'].astype(float)

cat = merged.select_dtypes(include = 'object').columns.tolist()
num = ['duration', 'director_facebook_likes', 'cast_total_facebook_likes', 'budget']

### Data standardization (numerical variables)
# define the scaler
scaler=StandardScaler()
# make a copy of the original dataframe
merged_scaled=merged.copy()
# scale the numeric features
merged_scaled[num] = scaler.fit_transform(merged_scaled[num])


### Dummification (categorical variables)
catPre=['content_rating','country']
merged_dummies = pd.get_dummies(merged_scaled,columns=catPre, drop_first = True)

### Create bins for the IMDb score
label_bins = ["0-5", "5-6", "6-7", "7-10"]
merged_dummies["imdb_score_bins"]  = pd.cut(merged_dummies['imdb_score'], [0, 5, 6, 7, 10], labels=label_bins)
merged_dummies.drop(['imdb_score'], inplace=True, axis=1)


## ii. Modeling
X = merged_dummies.drop(columns='imdb_score_bins')
y = merged_dummies['imdb_score_bins']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Logistic regression
logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

### Random Forest Classifier
#Instantiate model with 100 decision trees
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth=4, min_samples_leaf=5, max_leaf_nodes=8)
#Train the model on training data
rfc.fit(X_train, y_train)
#Predictions
rfcpred = rfc.predict(X_test)


## iii. Predictions
df_y_test=pd.DataFrame(y_test)
df_X_test=pd.DataFrame(X_test)

XY_test=pd.merge(X_test, y_test, left_index=True, right_index=True)

df_rfcpred = pd.DataFrame(rfcpred)
df_rfcpred = df_rfcpred.rename(columns={0: "pred bin"})

XY_test = XY_test.reset_index()

predictions=pd.merge(XY_test, df_rfcpred, left_index=True, right_index=True)

print(predictions)

#4. Convert the dataset of results back to a Spark DataFrame
# df_pred = createDataFrame(predictions)

#5. Write this DataFrame to the same S3 bucket dmacademy-course-assets under the prefix 
#   vlerick/<your_name>/ as JSON lines. It is likely Spark will create multiple files there. 
#   That is entirely normal and inherent to the distributed processing character of Spark.

# import boto3
# import io

# # Set the S3 bucket name and prefix
# BUCKET = "dmacademy-course-assets"
# prefix = "vlerick/Wannes/"

# # Get the S3 client
# s3 = boto3.client("s3")

# # Write the DataFrame to S3 as JSON lines
# df_pred.to_json(f"s3://{BUCKET}/{prefix}", orient="records", lines=True)

# #################
# #Creating Session using Boto3

# session = boto3.Session(
# aws_access_key_id='<key ID>',
# aws_secret_access_key='<secret_key>'
# )
 
# #Create s3 session with boto3

# s3 = session.resource('s3')
 
# json_buffer = io.StringIO()
 
# # Create dataframe and convert to pandas
# df = spark.range(4).withColumn("organisation", lit("stackoverflow"))
# df_p = df.toPandas()
# df_p.to_json(json_buffer, orient='records')
 
# #Create s3 object
# object = s3.Object('<bucket-name>', '<JSON file name>')
 
# #Put the object into bucket
# result = object.put(Body=json_buffer.getvalue())
# ###############