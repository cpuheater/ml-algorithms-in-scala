package com.cpuheater.ml

import com.cpuheater.ml.supervised.{LinearRegression, LogisticRegression}
import com.cpuheater.ml.util.{BagOfWordsTransformer, TestSupport}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

class TextClassificationSpec  extends TestSupport{

  private val english = "english"
  private val spanish = "spanish"
  private val sentences = List(spanish ->"Socorrimos a los heridos",
    spanish -> "He gastado todo el dinero en este coche",
    spanish -> "Gran Vía es la calle perpendicular a la calle Verde",
    english -> "He is coming here tomorrow",
    english -> "It is a Beautiful Day",
    english -> "My cats jump on my back")


  it should "text classification using logistic regression" in {

    val transformer = new BagOfWordsTransformer()
    transformer.fit(sentences)

    val (y, x) = transformer.transform(sentences)

    val lr = 0.001f
    val iterations = 1000

    val model = new LogisticRegression()
    model.fit(x, y, lr, iterations)

    val spanishSentence = transformer.transform(sentences.filter(_._1 == spanish).head._2)
    val englishSentence = transformer.transform(sentences.filter(_._1 == english).head._2)


    val pred1 = model.predict(spanishSentence)
    val pred2 = model.predict(englishSentence)

    println(s"Sentence ${spanish} belongs to class ${pred1}")
    println(s"Sentence ${english} belongs to class ${pred2}")

  }


}
