import random
random.seed(123)


# Uploading Data


income_data = spark.read("header","true").option("inferSchema","true").csv("adult.csv")

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data Preparation for spark Machine Learning


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


assembler = VectorAssembler(inputCols = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'], outputCol = 'features')
encoder = StringIndexer(InputCol = "income", outputCol = "label")
process.pipeline = Pipeline(stages = [assembler, encoder])

final_design = process_pipeline.fit(income_data)transform(income_data).select("features", "label")

train, validation, test = final_designrandmSplt([0.75, 0.15, 0.1], seed = 123)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Model Selecting and trying to find best 

from pyspark.ml.evaluation import MulticlassClassificationEvaulator
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(maxDepth = 13, numTrees = 5, seed = 123)
rfModel = rf.fit(train)

predictions = rfModel.transform(validation)
evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
accuracy = evaulator.evaulate(predictions)

print(accuracy)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training data with different depths ... 13 times (1 to 13)

from pyspark.ml.evaluation import MulticlassClassificationEvaulator
from pyspark.ml.classification import RandomForestClassifier

depths = [1, 3, 5, 7, 10,13, 15, 17,20, 23, 25, 27, 30]
accuracies = []
for  depth in depths:
	rf = RandomForestClassifier(maxDepth = depth, seed = 123)
	rfModel = rf.fit(train)

	predictions = rfModel.transform(validation)
	evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
	accuracy = evaulator.evaulate(predictions)

	accuracies.append(accuracy)

print(accuracies)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 #Visualization of the accuracy relationships of models 
import plotly
import plotly.graph_objs as go	
  	  
def plot(plot_dic, height = 1000, width = 1000, **kwargs):
	kwargs['output_type'] = 'div'
	plot_str = plotly.pffline.plot(plot_dic, **kwargs)
	print('%%angular <div style = "height : %ipx; width: %spx"> %s </div>' % (height, width, plot_str))

trace = go.Scatter(
	x = depths,
	y = accuracies
)

data = [trace]

plot(data)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Training data with different depths ... 13 times (1 to 14)

from pyspark.ml.evaluation import MulticlassClassificationEvaulator
from pyspark.ml.classification import RandomForestClassifier

trees = range(1, 14)
accuracies2 = []
for tree in trees:
	rf = RandomForestClassifier(numTrees = tree, seed = 123)
	rfModel = rf.fit(train)

	predictions = rfModel.transform(validation)
	evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
	accuracy = evaulator.evaulate(predictions)

	accuracies2.append(accuracy)

print(accuracies2)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Visualization of the accuracy relationships of models 

import plotly
import plotly.graph_objs as go

def plot(plot_dic, height = 1000, width = 1000, **kwargs):
	kwargs['output_type'] = 'div'
	plot_str = plotly.offline.plot(plot_dic, **kwargs)
	print(' %%angular <div style = "height: %ipx; width: %spx"> %s </div>' % (height, width, plot_str))

trace = go.Scatter(
	x = trees,
	y = accuracies2
)


data = [trace]

plot(data)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Training data with different depths ... 13 times (1 to 14) with 2 dimensional spaces

depths2 = [1, 3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 30]
trees2 = range(1, 14)
accuracies3 = []
looped_depths = []
looped_trees = []
for depth in depths:
	for tree in trees:
		rf = RandomForestClassifier(maxDepth = depth, numTrees = tree, seed = 123)
		rfModel = rf.fit(train)


		predictions = rfModel.transform(validation)
		evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
		accuracy = evaulator.evaulate(predictions)

		accuracies3.append(accuracy)
		looped_depths.append(depth)
		looped_trees.append(tree)

		to_print = "Accuracy: " + str(accuracy) + " ||| Depth: " + str(depth) + " ||| Trees: " + str(tree) + "\n"

		print(to_print)

print(accuracies3)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To simplify the following steps using this equation ...
# Each random draw has a 'q'  percent chance of landing in the required range
# Therefore, the probability that all random draws miss is (1-q)**n percent
# This means that the probability that at least one draw hits is 1-(1-q)**n 
# We require at least 'p' percent chance that we get within the required range 
# Therefore, our probability of success (1-(1-q)**n) must be greater than 'p'

import pandas as pd 


side_by_side = [accuracies3, looped_depths, looped_trees]
side_by_side = pd.DataFrame({'accuracies3' : accuracies3})
side_by_side['looped_depths'] = looped_depths
side_by_side['looped_trees'] = looped_trees

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Showing the result with 3 dimensions

import plotly
import plotly.graph_objs as go

def plot(plot_dic, height = 1000, width = 1000, **kwargs):
	kwargs['output_type'] = 'div'
	plot_str = plotly.offline.plot(plot_dic, **kwargs)
	print(' %%angular <div style = "height: %ipx; width: %spx"> %s </div>' % (height, width, plot_str))


data = [
	go.Mesh3d(
		z = side_by_side['accuracies3'],
		x = side_by_side['looped_depths'],
		y = side_by_side['looped_trees'],
		colorbar = go.ColorBar(
			title = 'Accuracy'
		),
		intensity = side_by_side['accuracies3']
	)
]

plot(data)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sorting our results by accuracy

sorted = side_by_side.sort_values(by = ['accuracies3'], ascending = False)
z.show(sorted.head(n-9))


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We just looked the first 60 results and then I scored them.


random_parameters = side_by_side.sample(n = 60).reset_index(drop = True)

accuracies4 = []
for i in range(0,60):
	depth = random_parameters['looped_depths'] [i]
	tree = random_parameters['looped_trees'] [i]


	rf = RandomForestClassifier(maxDepth = depth, numTrees = tree, seed = 123)
	rf.rfModel = rf.fit(train)


	predictions = rf.rfModel.transform(validation)
	evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
	accuracy = evaulator.evaulate(predictions)


	accuracies4.append(accuracy)

	to_print = "Accuracy: " + str(accuracy) + " ||| Depth: " + str(depth) + " ||| Trees: " + str(tree) + "\n"

	print(to_print)

random_parameters['accuracies4'] = accuracies4

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Showing the results as a 3d graph for evaluation

import plotly
import plotly.graph_objs as go

def plot(plot_dic, height = 1000, width = 1000, **kwargs):
	kwargs['output_type'] = 'div'
	plot_str = plotly.offline.plot(plot_dic, **kwargs)
	print(' %%angular <div style = "height: %ipx; width: %spx"> %s </div>' % (height, width, plot_str))


data = [
	go.Mesh3d(
		z = random_parameters['accuracies4'],
		x = random_parameters['looped_depths'],
		y = random_parameters['looped_trees'],
		colorbar = go.ColorBar(
			title = 'Accuracy'
		),
		intensity = random_parameters['accuracies4']
	)
]

plot(data)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the old data results 

z.show(sorted.head(n-9))

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the last data results 

sorted = random_parameters.sort_values(by=['accuracies4'], ascending = False)
z.show(sorted.head(n-1))

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Best model finding with bayesianoptimization algorithm 
from bayes_opt import BayesianOptimization

accuracies5 = []
depths3 = []
trees3 = []
def black_box_function(depth, tree):
	depth = int(round(depth))
	tree = int(round(tree))

	rf = RandomForestClassifier(maxDepth = depth, numTrees = tree, seed = 123)
	rfModel = rf.fit(train)


	predictions = rfModel.transform(validation)
	evaulator = MulticlassClassificationEvaulator(metricName = "accuracy")
	accuracy = evaulator.evaulate(predictions)


	accuracies5.append(accuracy)
	depths3.append(depth)
	trees3.append(tree)

	return accuracy

pbounds = {'depth': (1, 30), 'tree': (1, 13)}

optimizer = BayesianOptimization(
	f-black_box_function,
	pbounds = pbounds,
	random_state = 1,
)


optimizer.maximize(
	init_points = 10,
	n_iter = 60,
)


