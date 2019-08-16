import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object RandomForest {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession.builder()
      .appName("Activity Pred")
      .getOrCreate()

    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    val trainSet = spark.read.option("inferSchema", "true").option("header", "true").csv(args(0) + "/train.csv")
    val testSet = spark.read.option("inferSchema", "true").option("header", "true").csv(args(0) + "/test.csv")

    val featureCols = trainSet.columns.slice(0,trainSet.columns.size-2)

    val indexer = new StringIndexer().setInputCol("Activity").setOutputCol("activityIndex")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(100)
    val
    numTrees = 1000 // Use more in practice.
    val featureSubsetStrategy = "auto"
    val RandomForestModel1 = new RandomForestClassifier().setFeaturesCol("pcaFeatures").setLabelCol("activityIndex")
      .setFeatureSubsetStrategy(featureSubsetStrategy).setNumTrees(numTrees)
    val pipeline = new Pipeline().setStages(Array(indexer, assembler, pca, RandomForestModel1))

    val pipeFit = pipeline.fit(trainSet)
    val RanResult = pipeFit.transform(testSet)

    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("activityIndex")
    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(RanResult)

    sparkContext.parallelize(Array(accuracy.toString())).saveAsTextFile(args(0) + "/output_accuracy")



  }
}
