// Databricks notebook source
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// COMMAND ----------

// DBTITLE 1,Read DataSet
val trainSet = spark.read.option("inferSchema","true").option("header","true").csv("/FileStore/tables/train.csv")
val testSet = spark.read.option("inferSchema","true").option("header","true").csv("/FileStore/tables/test.csv")

// COMMAND ----------

println(s"No. of columns in Training set: ${trainSet.columns.size}")
println(s"Training Data points count: ${trainSet.count}")
println(s"Test Data points count: ${testSet.count}")

// COMMAND ----------

// DBTITLE 1,Feature Hasher
val featureCols = trainSet.columns.slice(0,trainSet.columns.size-2)
// var Array(red, _) = trainSet.randomSplit(Array(0.2, 0.8))
val hasher = new FeatureHasher().setInputCols(featureCols).setOutputCol("features")
// val transDf = hasher.transform(red)

// COMMAND ----------

// DBTITLE 1,String Indexer
val indexer = new StringIndexer()
  .setInputCol("Activity")
  .setOutputCol("label")
  .fit(trainSet)

// val indexed = indexer.transform(transDf)
//indexed.select("activityIndex").distinct.show()

// COMMAND ----------

// DBTITLE 1,Label's legend
val labelData = trainSet.select("Activity").distinct.toDF
val legend = indexer.transform(labelData).select("Activity", "label")
legend.show()

// COMMAND ----------

// DBTITLE 1,Logistic Regression
import org.apache.spark.ml.classification.LogisticRegression
val lr = new LogisticRegression()

// COMMAND ----------

val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .addGrid(lr.maxIter, Array(10, 20, 50))
  .build()

val lrPipeline = new Pipeline()
  .setStages(Array(hasher, indexer, lr))

val lrCv = new CrossValidator()
  .setCollectSubModels(true)
  .setEstimator(lrPipeline)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(10)
  //.setParallelism(200)

val lrCvModel = lrCv.fit(trainSet)

val lrBestModel = lrCvModel.bestModel

// COMMAND ----------

// DBTITLE 1,Save LR Model
lrCvModel.save("/saved-data/lrModel")

// COMMAND ----------

// DBTITLE 1,Load LR Model
val lrCvModel = CrossValidatorModel.load("/saved-data/lrModel")
val lrBestModel = lrCvModel.bestModel

// COMMAND ----------

// DBTITLE 1,LR-Evaluation
val lrTestResult = lrBestModel.transform(testSet).select("features", "probability", "label", "prediction")

val predictionAndLabels = lrTestResult.select("label", "prediction").map(x => (x.getDouble(0), x.getDouble(1))).rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
print("\nConfusion matrix:")
print("\n" + metrics.confusionMatrix.toString())

val accuracy = metrics.accuracy
print("\nSummary Statistics")
print(s"\nAccuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  print(s"\nPrecision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  print(s"\nRecall($l) = " + metrics.recall(l))
}

labels.foreach { l =>
  print(s"\nFalse Pos. Rate($l) = " + metrics.falsePositiveRate(l))
}

labels.foreach { l =>
  print(s"\nF1-Score($l) = " + metrics.fMeasure(l))
}

print(s"\nWeighted precision: ${metrics.weightedPrecision}")
print(s"\nWeighted recall: ${metrics.weightedRecall}")
print(s"\nWeighted F1 score: ${metrics.weightedFMeasure}")
print(s"\nWeighted false positive rate: ${metrics.weightedFalsePositiveRate}")
println("\n")

// COMMAND ----------

val lrTestResult = lrBestModel.transform(trainSet).select("features", "probability", "label", "prediction")

val predictionAndLabels = lrTestResult.select("label", "prediction").map(x => (x.getDouble(0), x.getDouble(1))).rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
print("\nConfusion matrix:")
print("\n" + metrics.confusionMatrix.toString())

val accuracy = metrics.accuracy
print("\nSummary Statistics")
print(s"\nAccuracy = $accuracy")

val labels = metrics.labels
labels.foreach { l =>
  print(s"\nPrecision($l) = " + metrics.precision(l))
}

labels.foreach { l =>
  print(s"\nRecall($l) = " + metrics.recall(l))
}

labels.foreach { l =>
  print(s"\nFalse Pos. Rate($l) = " + metrics.falsePositiveRate(l))
}

labels.foreach { l =>
  print(s"\nF1-Score($l) = " + metrics.fMeasure(l))
}

print(s"\nWeighted precision: ${metrics.weightedPrecision}")
print(s"\nWeighted recall: ${metrics.weightedRecall}")
print(s"\nWeighted F1 score: ${metrics.weightedFMeasure}")
print(s"\nWeighted false positive rate: ${metrics.weightedFalsePositiveRate}")
println("\n")

// COMMAND ----------


