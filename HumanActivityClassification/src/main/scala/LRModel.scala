import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{FeatureHasher, StringIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LRModel {

  def main(args: Array[String]): Unit = {
    
    //Creating spark session
    val spark = SparkSession.builder()
      .appName("Activity Pred")
      .getOrCreate()

    import spark.implicits._

    //Creating spark context
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    //Reading train and test dataset
    val trainSet = spark.read.option("inferSchema", "true").option("header", "true")
      .csv(args(0) + "/train.csv")
    val testSet = spark.read.option("inferSchema", "true").option("header", "true")
      .csv(args(0) + "/test.csv")

    //transforming the required features using FeatureHasher
    val featureCols = trainSet.columns.slice(0, trainSet.columns.length - 2)
    val hasher = new FeatureHasher().setInputCols(featureCols).setOutputCol("features")
    
    //Indexing the classification column from string to int - (0-5)
    val indexer = new StringIndexer()
      .setInputCol("Activity")
      .setOutputCol("label")
      .fit(trainSet)

    val labelData = trainSet.select("Activity").distinct.toDF
    val legend = indexer.transform(labelData).select("Activity", "label")

    //Defining a Logistic Regression model
    val lr = new LogisticRegression()
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.3, 0.1, 0.01))
      .addGrid(lr.maxIter, Array(10, 20, 50))
      .build()

    //Building a pipeline for the LR modeling
    val pipeline = new Pipeline()
      .setStages(Array(hasher, indexer, lr))

    //10-fold CrossValidation
    val cv = new CrossValidator()
      .setCollectSubModels(true)
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    //Traing the model
    val cvModel = cv.fit(trainSet)

    //Transforming the best LR model on test dataset
    val bestModel = cvModel.bestModel
    val testResult = bestModel.transform(testSet)
      .select(hasher.getOutputCol, "probability", indexer.getOutputCol, "prediction")

    //Building the output metrics string
    var buffer = new StringBuilder()

    legend.collect().foreach(x => buffer.append("\n%f %s".format(x.getDouble(1), x.getString(0))))

    val predictionAndLabels = testResult.select("label", "prediction")
      .map(x => (x.getDouble(0):Double, x.getDouble(1):Double)).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    buffer.append("\nConfusion matrix:")
    buffer.append("\n" + metrics.confusionMatrix.toString())

    val accuracy = metrics.accuracy
    buffer.append("\nSummary Statistics")
    buffer.append(s"\nAccuracy = $accuracy")

    val labels = metrics.labels
    labels.foreach { l =>
      buffer.append(s"\nPrecision($l) = " + metrics.precision(l))
    }

    labels.foreach { l =>
      buffer.append(s"\nRecall($l) = " + metrics.recall(l))
    }

    labels.foreach { l =>
      buffer.append(s"\nFPR($l) = " + metrics.falsePositiveRate(l))
    }

    labels.foreach { l =>
      buffer.append(s"\nF1-Score($l) = " + metrics.fMeasure(l))
    }

    buffer.append(s"\nWeighted precision: ${metrics.weightedPrecision}")
    buffer.append(s"\nWeighted recall: ${metrics.weightedRecall}")
    buffer.append(s"\nWeighted F1 score: ${metrics.weightedFMeasure}")
    buffer.append(s"\nWeighted false positive rate: ${metrics.weightedFalsePositiveRate}")
    
    //Saving the observed metrics onto a file
    sparkContext.parallelize(Seq(Seq(buffer.toString()))).saveAsTextFile(args(0) + "/output/metrics")

    //Saving the built model for future use
    cvModel.save(args(0) + "/savedModel/LR")
  }
}
