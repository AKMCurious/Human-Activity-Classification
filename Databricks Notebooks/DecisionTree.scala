// Databricks notebook source
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.DecisionTreeClassifier

// COMMAND ----------

// DBTITLE 1,Read Dataset
val trainSet = spark.read.option("inferSchema","true").option("header","true").csv("/FileStore/tables/train.csv")
val testSet = spark.read.option("inferSchema","true").option("header","true").csv("/FileStore/tables/test.csv")

// COMMAND ----------

// DBTITLE 1,Vector Assembler
val featureCols = trainSet.columns.slice(0,trainSet.columns.size-2)
// var Array(red, _) = trainSet.randomSplit(Array(0.2, 0.8))
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

// COMMAND ----------

// DBTITLE 1,String Indexer
val indexer = new StringIndexer()
  .setInputCol("Activity")
  .setOutputCol("label")
  .fit(trainSet)

// COMMAND ----------

// DBTITLE 1,Labels
val labelData = trainSet.select("Activity").distinct.toDF
val legend = indexer.transform(labelData).select("Activity", "label")
legend.show()

// COMMAND ----------

// DBTITLE 1,PCA
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(200)//.fit(df)

// COMMAND ----------

// DBTITLE 1,Decision Tree
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("pcaFeatures")

// COMMAND ----------

// DBTITLE 1,Training
val paramGrid = new ParamGridBuilder()
  .build()

val pipeline = new Pipeline()
  .setStages(Array(assembler, indexer, pca, dt))

val cv = new CrossValidator()
  .setCollectSubModels(true)
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)
  .setParallelism(200)

val cvModel = cv.fit(trainSet)

val bestModel = cvModel.bestModel

val model = pipeline.fit(trainSet)

// COMMAND ----------

// DBTITLE 1,Save DT Model
cvModel.save("/saved-data/dtModel")

// COMMAND ----------

// DBTITLE 1,Load DT Model
val cvModel = CrossValidatorModel.load("/saved-data/dtModel")
val bestModel = cvModel.bestModel

// COMMAND ----------

// DBTITLE 1,Evaluation and Metrics
val testResult = cvModel.transform(testSet)

val predictionAndLabels = testResult.select("label", "prediction")
      .map(x => (x.getDouble(0), x.getDouble(1))).rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
print("\nConfusion matrix:")
print("\n" + metrics.confusionMatrix.toString())

val accuracy = metrics.accuracy
print("\nSummary Statistics")
print(s"\nAccuracy = $accuracy")

// COMMAND ----------


