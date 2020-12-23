# Recommendation_System

- Recommender System using spark machine learning library

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




- Friend Recommendation using MapReduce

A) Write a MapReduce program in Hadoop that implements a simple “Mutual/Common friend list of two friends". The key idea is that if two people are friend then they have a lot of mutual/common friends. This program will find the common/mutual friend list for them.

For example, Alice’s friends are Bob, Sam, Sara, Nancy Bob’s friends are Alice, Sam, Clara, Nancy Sara’s friends are Alice, Sam, Clara, Nancy

As Alice and Bob are friend and so, their mutual friend list is [Sam, Nancy] As Sara and Bob are not friend and so, their mutual friend list is empty. (In this case you may exclude them from your output).

Input: Input files

    soc-LiveJournal1Adj.txt The input contains the adjacency list and has multiple lines in the following format: Hence, each line represents a particular user’s friend list separated by comma.
    userdata.txt The userdata.txt contains dummy data which consist of column1 : userid column2 : firstname column3 : lastname column4 : address column5: city column6 :state column7 : zipcode column8 :country column9 :username column10 : date of birth.

Here, is a unique integer ID corresponding to a unique user and is a comma-separated list of unique IDs corresponding to the friends of the user with the unique ID . Note that the friendships are mutual (i.e., edges are undirected): if A is friend with B then B is also friend with A. The data provided is consistent with that rule as there is an explicit entry for each side of each edge. So when you make the pair, always consider (A, B) or (B, A) for user A and B but not both.

Output: The output should contain one line per user in the following format: <User_A>, <User_B><Mutual/Common Friend List>

where <User_A> & <User_B> are unique IDs corresponding to a user A and B (A and B are friend). < Mutual/Common Friend List > is a comma-separated list of unique IDs corresponding to mutual friend list of User A and B.

Please generate/print the Mutual/Common Friend list for the following users.

(0,1), (20, 28193), (1, 29826), (6222, 19272), (28041, 28056)



B) Find friend pair(s) whose number of common friends is the maximum in all the pairs.

Output Format: <User_A>, <User_B><Mutual/Common Friend Number>



C) Use in-memory join at the Mapper to do the following. Given any two Users (they are friend) as input, output the list of the names and the date of birth (mm/dd/yyyy) of their mutual friends.

Note that the userdata.txt will be used to get the extra user information and cached/replicated at each mapper.

Output format: UserA id, UserB id, list of [names: date of birth (mm/dd/yyyy)] of their mutual Friends.

Sample Output: 1234 4312 [John:12/05/1985, Jane : 10/04/1983, Ted: 08/06/1982]



D) Use in-memory join at the Reducer to do the following:

For each user print User ID and maximum age of direct friends of this user.

Sample output:

User A, 60 where User A is the id of a user and 60 represents the maximum age of direct friends for this particular user. What to submit (i) Submit the source code via the eLearning website. (ii) Submit the output file for each question.



