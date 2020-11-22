import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

#create spark session
spark = SparkSession.builder.appName('Collaborative_Filtering').getOrCreate()

#read csv file ratings.dat and name columns
data_frame = pd.read_csv("ratings.dat", sep= "::", names = ['user_id','movie_id','ratings','timestamp'])
#drop column timestamp
data_frame = data_frame.drop(['timestamp'], axis = 'columns')

#convert to spark_dataframe
spark_dataframe = spark.createDataFrame(data_frame)

#split the data into training and test
(training,test)=spark_dataframe.randomSplit([0.7, 0.3])

#create als model
als=ALS(maxIter=10,regParam=0.09,rank=25,userCol="user_id",itemCol="movie_id",ratingCol="ratings",coldStartStrategy="drop",nonnegative=True)
#als=ALS(userCol="user_id",itemCol="movie_id",ratingCol="ratings",coldStartStrategy="drop",nonnegative=True)

#tune model
#param_grid = ParamGridBuilder().addGrid(als.rank,[12,20,25]).addGrid(als.maxIter,[10,15,20]).addGrid(als.regParam,[0.09, 0.17, 0.19]).build()

model=als.fit(training)

#evaluating the model using regression on metric root mean square error
evaluator=RegressionEvaluator(metricName="rmse",labelCol="ratings",predictionCol="prediction")

#Build CrossValidation
#tvs = TrainValidationSplit(estimator=als,estimatorParamMaps=param_grid,evaluator=evaluator)

#fiting the training data to model to train it


#best_model = model.bestModel

#run the model on test data
predictions=model.transform(test)

#calculating rmse using predictions
rmse=evaluator.evaluate(predictions)
print("RMSE="+str(rmse))
#RMSE=0.8909184124528494


#predictions.show(1000)
   