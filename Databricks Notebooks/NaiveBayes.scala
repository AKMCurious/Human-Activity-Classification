// Databricks notebook source
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression

// COMMAND ----------

// DBTITLE 1,Reading dataset
val trainDf = spark.read.option("header", "true").option("inferSchema","true").csv("/FileStore/tables/train.csv")
val testDf = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/test.csv")

// COMMAND ----------

// DBTITLE 1,Features columns
val featuresCols = trainDf.columns.slice(0, trainDf.columns.size - 2)

// COMMAND ----------

// DBTITLE 1,Feature Hasher
val hasher = new FeatureHasher().setInputCols(featuresCols).setOutputCol("features")

// COMMAND ----------

// DBTITLE 1,String Indexer
val indexer = new StringIndexer()
    indexer.setInputCol("Activity")
    indexer.setOutputCol("output")

// COMMAND ----------

// DBTITLE 1,Label data
val labelData = trainDf.select("Activity").distinct.toDF
val legend = indexer.fit(labelData).transform(labelData).select("Activity", "output")
legend.show()

// COMMAND ----------

// DBTITLE 1,Scale the output in 0 - 1 
val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

// COMMAND ----------

val naivebayes = new NaiveBayes().setLabelCol("output").setFeaturesCol("scaledFeatures")

// COMMAND ----------

// DBTITLE 1,Pipeline for training data
val pipeline = new Pipeline()
  .setStages(Array(hasher,indexer,scaler, naivebayes))

// COMMAND ----------

// DBTITLE 1,Creating a model for Naive bayes
val model = pipeline.fit(trainDf)

// COMMAND ----------

// DBTITLE 1,Testing results
val result = model.transform(testDf)

// COMMAND ----------

// DBTITLE 1,Multi class evaluator
val evaluator = new MulticlassClassificationEvaluator() 
evaluator.setLabelCol("output")
evaluator.setMetricName("accuracy")
val accuracy = evaluator.evaluate(result)
