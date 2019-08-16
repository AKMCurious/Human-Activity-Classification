name := "HumanActivityClassification"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.3"

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-library" % "2.11.12" % "provided",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)