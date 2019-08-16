// Databricks notebook source
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, VectorAssembler}

// COMMAND ----------

// DBTITLE 1,Imports for Random Forest
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.ml.feature.PCA
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

// COMMAND ----------

// DBTITLE 1,filePaths
val filepath_train = "/FileStore/tables/train.csv"
val filepath_test = "/FileStore/tables/test.csv"

// COMMAND ----------

// DBTITLE 1,Load Data
//val trainDF = spark.read.option("header","true").csv(filepath_train)
//val testDF = spark.read.option("header", "true").csv(filepath_test)
val trainSet = spark.read.option("inferSchema","true").option("header","true").csv(filepath_train)
val testSet = spark.read.option("inferSchema","true").option("header","true").csv(filepath_test)

// COMMAND ----------

val featureCols = trainSet.columns.slice(0,trainSet.columns.size-2)

// COMMAND ----------

// DBTITLE 1,Building a Pipeline for Random Forest Classifier
//val hasher = new FeatureHasher().setInputCols(featureCols).setOutputCol("features")
val indexer = new StringIndexer().setInputCol("Activity").setOutputCol("activityIndex")
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(100)//.fit(df)
//val treeStrategy = Strategy.defaultStrategy("Classification")
val numTrees = 1000 // Use more in practice.
val featureSubsetStrategy = "auto"
val RandomForestModel1 = new RandomForestClassifier().setFeaturesCol("pcaFeatures").setLabelCol("activityIndex").setFeatureSubsetStrategy(featureSubsetStrategy).setNumTrees(numTrees)
val pipeline = new Pipeline()
  .setStages(Array(indexer, assembler, pca, RandomForestModel1))

// COMMAND ----------

val pipeFit = pipeline.fit(trainSet)

// COMMAND ----------

val RanResult = pipeFit.transform(testSet)

// COMMAND ----------

// DBTITLE 1,Multi-Class Evaluator for Random Forest
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
val evaluator = new MulticlassClassificationEvaluator() 
evaluator.setLabelCol("activityIndex")
evaluator.setMetricName("accuracy")
val accuracy = evaluator.evaluate(RanResult)

// COMMAND ----------

// DBTITLE 0,Vector_Assembly For PCA
//import org.apache.spark.ml.feature.{PCA, VectorAssembler}

//val assembler = new VectorAssembler()
//      .setInputCols(testDF.schema.fields.map(_.name).slice(0,testDF.columns.size-2 ))
//      .setOutputCol("features")

//val df = assembler.transform(testDF)


// COMMAND ----------

// DBTITLE 0,PCA
//val pca = new PCA()
//      .setInputCol("features")
//      .setOutputCol("pcaFeatures")
//      .setK(200)
//      .fit(df)

// COMMAND ----------

// DBTITLE 0,PCA Transformation
//val result = pca.transform(df).select("pcaFeatures")

// COMMAND ----------

 val paramGrid = new ParamGridBuilder().build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

// COMMAND ----------

//SET spark.sql.broadcastTimeout = 1200
//import org.apache.spark.sql.functions.broadcast
//broadcast(spark.table("src")).join(spark.table("records"), "key").show()

// COMMAND ----------

val cvModel = cv.fit(trainSet)

// COMMAND ----------

val transformedModel = cvModel.transform(testSet)

// COMMAND ----------

val metrics = new MultiClassMetrics(transformedModel)
