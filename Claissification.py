#!/usr/bin/env python2.7
#coding: utf-8 andre1



from __future__ import print_function
from __future__ import division
# pyspark package
from pyspark import SparkContext
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.sql import Row
from pyspark.mllib.classification import SVMWithSGD
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.classification import LabeledPoint
# ML package
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# regular package
import numpy as np
import sys, os, json, re, subprocess, itertools

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

def load_features_value(path):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> extracting values from features file..")
	for feature_file in os.listdir(path):
		# handling label
		label = re.sub(r'[0-9]', '', feature_file)
		label = label[:-9].strip('_')
		# getting values
		for line in open(path + feature_file, "r"):
			values = line.strip('[]').split(',')
			values = [float(x) for x in values]
			# handling features
		features_value_list.append([label,values,feature_file])
	return features_value_list

def load_features_df_1vs1(features_value_list):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> creating dataframes from features values for 1vs1 classifier..")
	for feature in features_value_list:
		features_row = Row(label=feature[0], features=feature[1], images=feature[2])
		features_row_list.append(features_row)
	print(">>>>> creating rdd from features-row list..")
	features_rdd = sc.parallelize(features_row_list)
	#on distribut le RDD sans inquer le nombre de partition
	print(">>>>> creating dataframe from rdd..")
	features_df = spark.createDataFrame(features_rdd)
	
	return features_df

def load_features_df_1vsAll(features_value_list,class1):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> creating dataframes from features values for 1vs1 classifier..")
	for feature in features_value_list:
		if feature[0] == class1:
			features_row = Row(label=feature[0], features=feature[1], images=feature[2])
		else:
			features_row = Row(label="All", features=feature[1], images=feature[2])
		features_row_list.append(features_row)
	print(">>>>> creating rdd from features-row list..")
	features_rdd = sc.parallelize(features_row_list)
	#on distribut le RDD sans inquer le nombre de partition
	print(">>>>> creating dataframe from rdd..")
	features_df = spark.createDataFrame(features_rdd)
	return features_df

def convert_labels(train_features_df,test_features_df):
	# [CONVERT LABELS] Convert string labels to floats with Estimator
	label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
	# on cree lestimator qui va nous servir a transformer
	label_indexer_transformer = label_indexer.fit(train_features_df)
	# on cree notre transformateur
	train_features_df = label_indexer_transformer.transform(train_features_df)
	test_features_df = label_indexer_transformer.transform(test_features_df)
	# transforme la colone label en flotant et atribut aux labels une valeur entre 0.0 et 1.0
	
	return train_features_df, test_features_df

def convert_labels_All(train_features_df,test_features_df,test_features_df2):
	# [CONVERT LABELS] Convert string labels to floats with Estimator
	label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
	# on cree lestimator qui va nous servir a transformer
	label_indexer_transformer = label_indexer.fit(train_features_df)
	# on cree notre transformateur
	train_features_df = label_indexer_transformer.transform(train_features_df)
	test_features_df = label_indexer_transformer.transform(test_features_df)
	test_features_df2 = label_indexer_transformer.transform(test_features_df2)
	# transforme la colone label en flotant et atribut aux labels une valeur entre 0.0 et 1.0
	
	return train_features_df, test_features_df, test_features_df2


def training(class1,class2):
	print("\n" + "+"*40)
	print("+++++ Training for %s vs. %s classifier" % (class1,class2))
	print("+"*40 + "\n")
	# grid search on SVMWithSGD model parameters - train model for each combination
	# return best model parameters
	from pyspark.mllib.classification import SVMWithSGD

	# model parameters
	model_number = 0
	numIters = [10,50,100,300]
	stepSizes = [1]
	regParams = [0.01]
	best_model_number = None
	best_model = None
	best_prediction_df = None
	best_accuracy = 0
	best_trainErr = 0
	best_errors = 0
	best_iter = None
	best_stepSize = None
	best_regParam = None
	# grid training
	for numIter,stepSize,regParam in itertools.product(numIters,stepSizes,regParams):
		model_number += 1
		print(">>>>> Building model #%i.." % (model_number))
		print(train_features_lp)
		model = SVMWithSGD.train(train_features_lp, numIter, stepSize, regParam)
		
		# [TEST] Guess labels on test data
		print(">>>>> Testing model #%i.." % (model_number))
		prediction_lp = test_features_lp.map(lambda p : Row(label_index_predicted=model.predict(p.features), label_index=p.label, features=p.features))
		

		# retourne un rdd et ne preserve pas le partitioning par defaut
		prediction_df = spark.createDataFrame(prediction_lp)
		err=prediction_df.filter(prediction_df.label_index_predicted != prediction_df.label_index)
		print("show err")
		#err.show(80)
		if class2 == "All": 
			prediction_lp2 = test_features_lp2.map(lambda p : Row(label_index_predicted=model.predict(p.features), label_index=p.label, features=p.features))
			prediction_df_All = spark.createDataFrame(prediction_lp2)
			err_All=prediction_df_All.filter(prediction_df_All.label_index_predicted != prediction_df_All.label_index)
			print("show err_All")
			accuracy_All = prediction_df_All.filter(prediction_df_All.label_index_predicted == prediction_df_All.label_index).count() / float(prediction_df_All.count())
			trainErr_All = prediction_df_All.filter(prediction_df_All.label_index_predicted != prediction_df_All.label_index).count() / float(prediction_df_All.count())
			errors_All = prediction_df_All.filter(prediction_df_All.label_index_predicted != prediction_df_All.label_index).count()


		image_indexer = [StringIndexer(inputCol="images", outputCol="images_index")]
		pipeline = Pipeline(stages=image_indexer)
		image_features_indexer = pipeline.fit(test_features_df).transform(test_features_df)
		#print ("test_features_df")
		#test_features_df.show()
		#image_indexer = StringIndexer(inputCol="images", outputCol="images_index")
		#image_indexer_transformer = image_indexer.fit(test_features_df)
		#image_features_df = image_indexer_transformer.transform(test_features_df)

		image_features_lp = image_features_indexer.rdd.map(lambda row: LabeledPoint(row.images_index, row.features))
		image_map_lp = image_features_lp.map(lambda image_map : Row(images_index=image_map.label, features=image_map.features))
		image_features_df = spark.createDataFrame(image_map_lp)
		#sparkConf.set("spark.sql.crossJoin.enabled", "true")
		err2 = image_features_df.join(err, image_features_df.features == err.features)

		#err2.show()

		#err2.take(2)
		err3 = image_features_indexer.join(err2, image_features_indexer.images_index == err2.images_index)
		
		#err3.show()
		err3.select(func.col("images").alias("images non reconnus")).show(30, truncate = False)


		#fichiercomplex.limit(8).show()
		#test_features_df.printSchema()
		#prediction_df.limit(2).show()
		# [EVALUATION] the model on training data
		print(">>>>> Evaluating model #%i.." % (model_number))
		accuracy = prediction_df.filter(prediction_df.label_index_predicted == prediction_df.label_index).count() / float(prediction_df.count())
		trainErr = prediction_df.filter(prediction_df.label_index_predicted != prediction_df.label_index).count() / float(prediction_df.count())
		errors = prediction_df.filter(prediction_df.label_index_predicted != prediction_df.label_index).count()
		# Is it the best model yet?
		if accuracy > best_accuracy:
			best_model_number = model_number
			best_model = model
			best_prediction_df = prediction_df
			best_accuracy = accuracy
			best_trainErr = trainErr
			best_errors = errors
			best_stepSize = stepSize
			best_regParam = regParam
			best_numIter = numIter
		# [RESULTS]
		print("""
		|Model #%i
		|Model trained with (numIter: %.2f, stepSize = %.2f, regParam = %.2f)
		|Model has accuracy of %.3f (errors: %i / training error: %.3f) on test
		""" % (model_number,numIter,stepSize,regParam,accuracy,errors,trainErr))
		if class2 == "All":
			print("""
			|Model_All #%i
			|Model_All trained with (numIter: %.2f, stepSize = %.2f, regParam = %.2f)
			|Model_All has accuracy of %.3f (errors: %i / training error: %.3f) on test
			""" % (model_number,numIter,stepSize,regParam,accuracy_All,errors_All,trainErr_All))
		

	# [RESULTS]
	# Display results for best model
	print("""
	|Model #%i is the best model
	|The best model was trained with (numIter: %.2f, stepSize = %.2f, regParam = %.2f)
	|The best model has accuracy of %.3f (errors: %i / training error: %.3f) on test
	""" % (best_model_number,best_numIter,best_stepSize,best_regParam,best_accuracy,best_errors,best_trainErr))
	# Add entry into best_models dictionnary
	best_models[("%s_vs._%s" % (class1,class2))] = { "Accuracy":best_accuracy,"numIter":best_numIter,"stepSize":best_stepSize,"regParam":best_regParam}

def main():
	# parameters
	features_dir = sys.argv[1]
	features_dir_all = sys.argv[2]
	global train_features_lp
	global test_features_lp
	global test_features_lp2
	global best_models
	global features_value_list
	global test_features_df
	features_value_list = []
	best_models = {}
	classes = []
	for feature_file in os.listdir(features_dir):
		new_class = re.sub(r'[0-9]', '', feature_file)
		new_class = new_class[:-9].strip('_')
		classes.append(new_class)
	classes = sorted(list(set(classes)))
	classes_dup = classes
	# [FEATURES EXTRACTION]
	# subprocess.call(["python", "features_extract.py"])

	# [LOADING FEATURE VALUES] loading featuresvalues into dictionnary
	print(">>>>> Loading features values into list of rows..")
	features_value_list = load_features_value(features_dir)
	features_value_list2 = load_features_value(features_dir)

	# [CLASSIFIER SELECTION] Selecting classifiers (1vs1, 1vsAll)
	# 1vs1 classifiers
	for class1 in classes:
		class2_set = [x for x in classes_dup]
		del class2_set[0:(classes.index(class1)+1)]
		print("classes")
		print(classes)
		print("class2_set")
		print(class2_set)
		for class2 in class2_set:
			print(">>>>> Building dataframes for classifier %s vs. %s.." % (class1,class2))
			# [LOADING FEATURES] loading features values into dataframe
			print("_____ Loading features values into main dataframe")
			features_df = load_features_df_1vs1(features_value_list)

			print("_____ Filtering data within dataframe")
			features_classifier_df = features_df.filter((features_df.label == class1) | (features_df.label ==  class2))
			# filtre le RDD pour ne laisser que les classe 1 et 2
			# [SPLIT DATA] Split data into train & test
			print("_____ Spliting data into training & test data..")
			train_features_df, test_features_df = features_classifier_df.randomSplit([0.5, 0.50])
			# cree deux variable 0.8 et 0.2 sans changer ou melanger le partionement (ici la valeur du second paramettre seed nest pas indquer)
			
			train_count = train_features_df.count()
			
			test_count = test_features_df.count()
			print("%i training data" % (train_count))
			print("%i testing data" % (test_count))
			# [CONVERET LABELS] Convert string labels into floats with an estimator
			print("_____ Converting string labels into floats with an estimator..")
			train_features_df, test_features_df = convert_labels(train_features_df,test_features_df)
			# [CONVERT INTO LABELDPOINTS]
			print(">>>>> Converting dataframe into labelpoint rdd..")
			
			train_features_lp = train_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
			
			
			test_features_lp = test_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))

			# [BUILD MODEL] Learn classifier on training data
			print(">>>>> Training classifier..")
			training(class1,class2)
	
	# 1vsAll classifiers
	print("1vsALL---------------------------------------------------------------------------------------")
	print("classes")
	print(classes)
	print("class2_set")
	for class1 in classes:
		print(">>>>> Building dataframes for classifier %s vs. All.." % (class1))
		# [LOADING FEATURES] loading features values into dataframe
		print("_____ Loading features values into main dataframe")
		features_df = load_features_df_1vsAll(features_value_list,class1)
		features_df2 = load_features_df_1vsAll(features_value_list2,class1)
		# [SPLIT DATA] Split data into train & test
		print("_____ Spliting data into training & test data..")
		train_features_df, test_features_df = features_df.randomSplit([0.8, 0.20])
		train_features_df2, test_features_df2 = features_df2.randomSplit([0.5, 0.50])

		train_count = train_features_df.count()

		test_count = test_features_df.count()
		test_count2 = test_features_df2.count()
		print("%i training data" % (train_count))
		print("%i testing data" % (test_count))
		print("%i testing All data" % (test_count))
		# [CONVERET LABELS] Convert string labels into floats with an estimator
		print("_____ Converting string labels into floats with an estimator..")
		train_features_df, test_features_df, test_features_df2 = convert_labels_All(train_features_df,test_features_df, test_features_df2)
		# [CONVERT INTO LABELDPOINTS]
		print(">>>>> Converting dataframe into labelpoint rdd..")
		train_features_lp = train_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
		test_features_lp = test_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
		test_features_lp2 = test_features_df2.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
		# [BUILD MODEL] Learn classifier on training data
		print(">>>>> Training classifier..")
		training(class1,"All")


	# [OUTPUT]
	# For each classifier, send model parameters in best_classifiers.json
	print(">>>>> Sending best model information to \"best_classifiers.json\"..")
	with open("./output/best_classifiers.json", "w") as out:
		json.dump(best_models, out)


	# hang script to tune it with Spark Web UI (available @ http://localhost:4040)
	raw_input("press ctrl+c to exit")

if __name__ == "__main__":
	main()
