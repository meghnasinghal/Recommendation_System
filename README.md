# Recommendation_System

1. Using spark machine learning library spark-mlib, use KMeans to cluster the movies using the ratings given by the user, that is, use the item-user matrix from itemusermat File provided as input to your program.

Dataset description.
Dataset: Itemusermat File.
The itemusermat file contains the ratings given to each movie by the users in Matrix format. The file contains the ratings by users for 1000 movies.
Each line contains the movies id and the list of ratings given by the users. 
A rating of 0 is used for entries where the user did not rate a movie.
From the sample below, user1 did not rate movie 2, so we use a rating of 0.
A sample Itemusermat file with the item-user matrix is shown below.

	user1	user2
movie1	4	3
movies2	0	2
Set the number of clusters (k) to 10
Your Scala/python code should produce the following output:

•	For each cluster, print any 5 movies in the cluster. Your output should contain the movie_id, movie title, genre, and the corresponding cluster it belongs to. Note: Use the movies.dat file to obtain the movie title and genre.

         For example
         cluster: 1
         123, Star Wars, sci-fi 
         
         
2. Use Collaborative filtering find the accuracy of ALS model accuracy. Use ratings.dat file. It contains User id :: movie id :: ratings :: timestamp.  Your program should report the accuracy of the model.

For details follow the link:    http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html  

Please use 70% of the data for training and 30% for testing and report the accuracy of the model.


