import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans


data = pd.read_csv('itemusermat', sep = " ", header = None) #reading file using pandas

spark = SparkSession.builder.appName('KMeans').getOrCreate() #creating spark session

data = data.replace(0, np.NaN) # replacing 0 with Nan so that they are not considered while calcuating mean

data['mean'] = data.iloc[:, 1:].mean(axis=1)  # calculated the average rating for each row (axis = 1) excluding the first column entry

dataframe = data[[0,'mean']].copy() #create copy of data with movie id and mean

dataframe = dataframe.rename(columns = {0 : 'movie_id'}) #rename column 

spark_dataframe = spark.createDataFrame(dataframe) #convert pandas datafrmae to spark dataframe

assembler = VectorAssembler(inputCols=['mean'], outputCol="features") 
#Indicate the column 'mean' in the df dataframe we want to use as features and the we use the VectorAasembler to put the feature columns into a new column called features that contains all of these as an array.

data_set=assembler.transform(spark_dataframe) # create a spark dataframe using tranform operation 

#data_set.show()

#Training a k-means model
kmeans = KMeans().setK(10).setSeed(1) 
#setK sets number of clusters
#setSeed reduces randomization, to intialte the centroids

model = kmeans.fit(data_set) # fitting the datset to the model


predictions = model.transform(data_set) # This will add the prediction column to the dataframe

# By computing Silhouette score, evaluate clustering 
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
#Silhouette with squared euclidean distance = 0.6891373470755809

# Evaluate clustering by calculating squared Euclidean distance - Values closer to 1 indicate maximum separation 
cost = model.computeCost(data_set)
print("Within Set Sum of Squared Errors = " + str(cost))

#Printing the cluster centers
print("Cluster Centers: ")
list_centers=[]
centers = model.clusterCenters()
for center in centers:
    list_centers.append(center)
    print("center",center)
    
# Cluster Centers: 
# center [3.99209921]
# center [3.06554168]
# center [1.99916712]
# center [2.45713504]
# center [4.30349901]
# center [2.77251053]
# center [3.75459386]
# center [1.15096245]
# center [3.32190314]
# center [3.53640168]

pandasDF=predictions.toPandas() #converting the dataframe to pandas dataframe
centers = pd.DataFrame(list_centers)

#print(pandasDF.head(50))

#creating a list of dataframes based on index =  prediction/cluster = i (1 to 10)
l = []
for i in range(10):
	df  = pandasDF.loc[pandasDF['prediction'] == i].head(5) #filtering dataframe for cluster i and taking 5 values
	l.append(df[['movie_id', 'prediction']]) # adding dataframe to list and taking only movie id and prediction

result_df = pd.concat(l) #concating the list to a result dataframe
#print(result_df)


moviedf = pd.read_csv('movies.dat', sep = "::",header = None)
#read movie details in pandas dataframe

moviedf = moviedf.rename(columns = {0 : 'movie_id', 1: 'movie_title', 2: 'genre'})
#rename column names

#print(moviedf.head())

res = pd.merge(moviedf, result_df, on = 'movie_id').sort_values(by = ['prediction'])
#merging the result dataframe and merging with movie dataframe based on movie id and sorting result based on cluster/prediction

res.to_csv("result", index = False) #writing result to csv file
print(res.head(30))

