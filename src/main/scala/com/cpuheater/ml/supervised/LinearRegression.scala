package com.cpuheater.ml.supervised

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4s.Evidences.float

class LinearRegression {

   private var computedThetas: INDArray = _

   def fit(x: INDArray, y: INDArray, lr: Float, iters: Int): INDArray = {
      val bias = Nd4j.ones(x.rows(), 1)
      val xWithBias =  Nd4j.concat(1, bias, x)
      val thetas =  Nd4j.zeros(1, xWithBias.columns()).reshape(1, xWithBias.columns())
      computedThetas = computeGradient(xWithBias, y, thetas, lr, iters)
      computedThetas
   }

   def predict(x: INDArray): Float = {
      val bias = Nd4j.ones(x.rows(), 1)
      val xWithBias =  Nd4j.concat(1, bias, x)
      xWithBias.mmul(computedThetas.T).getFloat(0)
   }

   private def computeCost(x: INDArray, y: INDArray, thetas: INDArray): Float = {
      val cost = pow((x.mmul(thetas.T)) - y, 2)
      (cost.sum(0)/(2*cost.length)).getFloat(0)
   }

   private def computeGradient(x: INDArray, y: INDArray, thetas: INDArray, lr: Float, iterations: Int): INDArray ={
      val nbOfTrainingExamples = x.rows
      val computedThetas = (0 to iterations).foldLeft(thetas)({
         case (thetas, i) =>
            val error = x.mmul(thetas.T) - y
            val grad = error.T.mmul(x)/nbOfTrainingExamples
            val updatedThetas = thetas - grad*lr
            println(s"Iteration ${i} cost: ${computeCost(x, y, updatedThetas)}")

            updatedThetas
      })
      computedThetas
   }

}



