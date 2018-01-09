package com.cpuheater.ml.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class BagOfWordsTransformer {

  def fit(data: List[(String, String)]) = {
    val (labels, values) = data.unzip
    wordToIdx = values.flatMap(s => tokenize(normalize(s))).distinct.zipWithIndex.toMap
    labelToIdx = labels.distinct.zipWithIndex.toMap
  }

  def transform(s: String): INDArray = {
    val vector = Nd4j.create(1, wordToIdx.size)
    tokenize(normalize(s)).foldLeft(vector){
      case (accum, word) =>
        wordToIdx.get(word).map{ idx =>
          vector.putScalar(idx, 1)
        }.getOrElse(accum)
    }
  }

  def transform(data: List[(String, String)]) : (INDArray, INDArray) = {
    val (labels, features) = data.foldLeft(List[Float](), List[INDArray]()){
      case ((labels, features), (labelName, sentence)) =>
        val label = labelToIdx(labelName)
        val feature = transform(sentence)
        (label::labels, feature::features)
    }
    (Nd4j.create(labels.toArray).reshape(labels.size, -1), Nd4j.concat(0, features.toArray: _*))
  }


  private def normalize(s: String): String = s.toLowerCase
  private def tokenize(s: String): List[String] = s.split(" ").toList
  private var wordToIdx: Map[String, Int] = _
  private var labelToIdx: Map[String, Int] = _

}
