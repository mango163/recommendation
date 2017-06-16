name := "recommendation"

version := "1.0"

scalaVersion := "2.10.6"

lazy val root = (project in file("."))

libraryDependencies ++= Seq(
	"log4j" % "log4j" % "1.2.17",
	"commons-io" % "commons-io" % "2.5",
	"commons-codec" % "commons-codec" % "1.10",
	"org.apache.commons" % "commons-lang3" % "3.4",
	"org.apache.spark" % "spark-core_2.10" % "1.6.0",
	"org.apache.spark" % "spark-sql_2.10" % "1.6.0",
	"org.apache.spark" % "spark-streaming_2.10" % "1.6.0",
	"org.apache.spark" % "spark-mllib_2.10" % "1.6.0",
	"org.apache.spark" % "spark-streaming-kafka_2.10" % "1.6.0",
	"org.apache.hadoop" % "hadoop-client" % "2.6.0"
)
