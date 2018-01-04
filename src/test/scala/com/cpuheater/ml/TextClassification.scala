package com.cpuheater.ml

import com.cpuheater.ml.supervised.{LinearRegression, LogisticRegression}
import com.cpuheater.ml.util.{BagOfWordsVectorizer, TestSupport}
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



    val vectorizer = new BagOfWordsVectorizer()
    vectorizer.fit(sentences)

    val (y, x) = vectorizer.transform(sentences)

    val lr = 0.001f
    val iterations = 1000

    val model = new LogisticRegression()
    model.fit(x, y, lr, iterations)

    val spanishSentence = vectorizer.transform(sentences.filter(_._1 == spanish).head._2)
    val englishSentence = vectorizer.transform(sentences.filter(_._1 == english).head._2)


    val pred1 = model.predict(spanishSentence)
    val pred2 = model.predict(englishSentence)

    println(s"Sentence ${spanish} belongs to class ${pred1}")
    println(s"Sentence ${english} belongs to class ${pred2}")



  }


  private def buildVocabulary(vocab: Set[String]): Map[String, Int] = {
    vocab.zipWithIndex.toMap
  }



}
