package com.cpuheater.ml

import java.io.File
import java.util
import java.util.{ArrayList, Arrays, List}

import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator
import com.cpuheater.ml.supervised.{LinearRegression, LogisticRegression}
import com.cpuheater.ml.util.TestSupport
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, FileSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

class TextClassificationSpec  extends TestSupport{


  it should "text classification using logistic regression" in {

    val rootDir: File = new ClassPathResource("classification").getFile
    var iter: LabelAwareSentenceIterator = new LabelAwareFileSentenceIterator(rootDir)
    val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor())

    val vectorizer = new BagOfWordsVectorizer.Builder()
      .setMinWordFrequency(1)
      .setStopWords(new java.util.ArrayList[String]())
      .setTokenizerFactory(tokenizerFactory)
      .setIterator(iter)
      .allowParallelTokenization(false).build

    vectorizer.fit()
    println(vectorizer.getVocabCache.tokens())
    println(vectorizer.getVocabCache.numWords())

    val ndArray = vectorizer.transform("a la los los")
    println(ndArray)

    iter = new LabelAwareFileSentenceIterator(rootDir)
    val categoryToLabel = iter.currentLabels().toList.zipWithIndex.toMap

    val lb = mutable.ListBuffer[(INDArray, INDArray)]()

    while(iter.hasNext){
      val sentence = iter.nextSentence()
      val y = categoryToLabel(iter.currentLabel())
      val x = vectorizer.transform(sentence)
      lb += (x -> Nd4j.create(Array(y.toFloat)))
    }

    val x = Nd4j.concat(0, lb.map(_._1).toArray: _*)
    val y = Nd4j.concat(0, lb.map(_._2).toArray: _*)

    val lr = 0.1f
    val iterations = 1000

    val model = new LogisticRegression()
    model.fit(x, y, lr, iterations)

    val text1 = vectorizer.transform("a.")
    println(text1)
    /*
    val pred1 = model.predict(text2)

    println(s"Point${point1.toList} belongs to class ${pred1}")
    println(s"Point${point2.toList} belongs to class ${pred2}")
*/

  }

}
