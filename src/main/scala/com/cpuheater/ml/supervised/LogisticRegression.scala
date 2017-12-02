package com.cpuheater.ml.supervised

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.{log, pow, sigmoid}
import org.nd4s.Implicits._

class LogisticRegression {

  private var computedThetas: INDArray = _

  def fit(x: INDArray, y: INDArray, lr: Float, iters: Int): Unit = {
    val ones = Nd4j.ones(x.rows(), 1)
    val xWithBias =  Nd4j.concat(1, ones, x)
    val thetas =  Nd4j.zeros(xWithBias.columns(), 1).T
    computedThetas = computeGradient(xWithBias, y, thetas, lr, iters)
  }

  def predict(x: INDArray): Float = {
    val bias = Nd4j.ones(x.rows(), 1)
    val xWithBias =  Nd4j.concat(1, bias, x)
    if(sigmoid(xWithBias.mmul(computedThetas.T)).getFloat(0) >0.5)
      1
    else
      0
  }

  private def computeCost(x: INDArray, y: INDArray, thetas: INDArray): Float = {
    val output = sigmoid(x.mmul(thetas.T))
    val term1 = log(output).mul(-y)
    val term2 = log(output.rsub(1)).mul(y.rsub(1))
    Nd4j.clearNans(term2)
    val crossEntropy = term1.sub(term2).sumNumber().floatValue()/x.shape()(0)
    crossEntropy
  }

  private def computeGradient(x: INDArray, y: INDArray, thetas: INDArray, lr: Float, iterations: Int): INDArray ={
    val nbOfTrainingExamples = x.rows()
    val updatedTheta = (0 to iterations).foldLeft(thetas)({
      case (thetas, i) =>
        val error = sigmoid(x.mmul(thetas.T)) - y
        val grad = error.T.dot(x)  * lr/nbOfTrainingExamples
        val updatedThetas =  thetas - grad
        println(s"Iteration ${i} cost: ${computeCost(x, y, updatedThetas)}")
        updatedThetas
    })
    updatedTheta

  }



}
