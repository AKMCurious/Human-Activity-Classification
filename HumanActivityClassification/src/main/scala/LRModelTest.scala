import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object LRModelTest {
  def main(args: Array[String]): Unit = {
    //Creating spark session
    val spark = SparkSession.builder()
      .appName("Activity Pred")
      .getOrCreate()

    import spark.implicits._

    //Creating spark context
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    //Reading the test dataset
    val testSet = spark.read.option("inferSchema", "true").option("header", "true")
      .csv(args(0) + "/test.csv")

    //Loading the saved LR model
    val cvModel = CrossValidatorModel.load(args(0) + "/savedModel/LR")

    //Transforming the best model on the test dataset
    val bestModel = cvModel.bestModel
    val testResult = bestModel.transform(testSet)

    //Building output metrics 
    var buffer = new StringBuilder()
    val predictionAndLabels = testResult.select("label", "prediction")
      .map(x => (x.getDouble(0): Double, x.getDouble(1): Double)).rdd
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

    //saving output metrics in a file
    sparkContext.parallelize(Seq(Seq(buffer.toString()))).saveAsTextFile(args(0) + "/output/metrics")
  }
}
