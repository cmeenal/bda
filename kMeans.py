'''
Type in  first Terminal:
./Anaconda-2019.07-Linux-x86_64.sh
Press Enter. Type ‘yes’

Unzip spark file(apache-spark-*.zip) in home folder(spark-2.3.1-bin-hadoop2.7 here)

Open new terminal and type
(base) hadoop@d50110-ThinkCentre-M720t:~$ pwd
(base) hadoop@d50110-ThinkCentre-M720t:~$ jupyter notebook
 => it redirects to browser and open new tab of jupyter notebook homepage

Create a new folder . rename it(spark practical )
Create new notebook inside it(goto New>Python notebook)
CLICK + to insert new cell.
Run each cell in order to get output


Open new terminal (don’t close 1st terminal or jupyter notebook will close)
(base) hadoop@d50110-ThinkCentre-M720t:~$ pip install findspark

If there is permission error while installing then write command
hadoop@d50110-ThinkCentre-M720t:~$ chmod -R 777 spark-2.3.1-bin-hadoop2.7

'''
import os
os.environ["HADOOP_HOME"] = "/home/hadoop/hadoop2" !echo $HADOOP_HOME
os.environ["SPARK_HOME"] = "/home/usermeenal/spark-2.3.1-bin-hadoop2.7"
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-1.8.0-openjdk-amd64"

import findspark
findspark.init()

from pyspark.sql import SparkSession
spark=SparkSession.builder.getOrCreate()
spark

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
#load data
dataset = spark.read.format("libsvm").load("file:///home/usermeenal/Downloads/sample_kmeans_data.txt")
#train a k-means model
kmeans = KMeans().setK(2).setSeed(1)
model=kmeans.fit(dataset)
#make predictions
predictions=model.transform(dataset)
#evaluate clustering by computing Silhouette score
evaluator=ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared Euclidean distance =" + str(silhouette))
#show the result
centers=model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
	print(center)
	
dataset.show()

import pandas as pd
df=datatset.toPandas()
l = []
for i in range(0,len(df["features"])):
	l.append(df["features"][i][0])
df

import matplotlib.pyplot as plt
plt.scatter(l,df['label'],c='red')
plt.scatter(centers,centers,c="black",s=100)
plt.grid()
plt.xlabel("Label")
plt.ylabel("Features")
plt.show()
