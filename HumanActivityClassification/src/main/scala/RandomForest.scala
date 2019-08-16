import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object RandomForest {
  def main(args: Array[String]): Unit =
  {
    //Creating a spark session
    val spark = SparkSession.builder()
      .appName("Activity Pred")
      .getOrCreate()

    //Creating a spark context
    val sparkContext = spark.sparkContext
    sparkContext.setLogLevel("ERROR")

    //Reading train and test data
    val trainSet = spark.read.option("inferSchema", "true").option("header", "true").csv(args(0) + "/train.csv")
    val testSet = spark.read.option("inferSchema", "true").option("header", "true").csv(args(0) + "/test.csv")
    
    //Columns to use for dimensionality reduction
    val featureCols = trainSet.columns.slice(0,trainSet.columns.size-2)

    //Indexing the classification column (0-5)
    val indexer = new StringIndexer().setInputCol("Activity").setOutputCol("activityIndex")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    
    //Reducing dimentionality of features to 100 (from over 500)
    //Settled on 1000 for 'no of trees' and reduced dimensionality to 100 (PCA) by trial and error - optimizing accuracy
    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(100)    
    val numTrees = 1000 
    
    //Creating a RandomForestClassifier with subset strategy as 'auto' and 1000 trees
    val featureSubsetStrategy = "auto"
    val RandomForestModel1 = new RandomForestClassifier().setFeaturesCol("pcaFeatures").setLabelCol("activityIndex")
      .setFeatureSubsetStrategy(featureSubsetStrategy).setNumTrees(numTrees)
    
    //Building a pipeline
    val pipeline = new Pipeline().setStages(Array(indexer, assembler, pca, RandomForestModel1))

    //Training the model
    val pipeFit = pipeline.fit(trainSet)
    
    //Transforming the model on test dataset
    val RanResult = pipeFit.transform(testSet)

    //Calculating accuracy on test dataset
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("activityIndex")
    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(RanResult)

    //Saving the accuracy to a file
    sparkContext.parallelize(Array(accuracy.toString())).saveAsTextFile(args(0) + "/output_accuracy")

  }
}
