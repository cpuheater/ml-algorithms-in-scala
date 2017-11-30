package com.cpuheater.ml.supervised

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.nd4s.Evidences.float

class LinearRegression {

   private var computedThetas: INDArray = _

   def fit(x: INDArray, y: INDArray, lr: Float, iters: Int): INDArray = {
      val bias = Nd4j.onesLike(x)
      val xWithBias =  Nd4j.concat(1, bias, x)
      val thetas =  Nd4j.zeros(1, xWithBias.columns()).reshape(1, xWithBias.columns())
      computedThetas = computeGradient(xWithBias, y, thetas, lr, iters)
      computedThetas
   }

   def predict(x: INDArray): Float = {
      x.mmul(computedThetas.T).getFloat(0)
   }

   private def computeCost(x: INDArray, y: INDArray, thetas: INDArray): Float = {
      val r = pow((x.mmul(thetas.T)) - y, 2)
      val r2 = r.sum(0)/(2*r.length)
      r2.getFloat(0)
   }


   private def computeGradient(x: INDArray, y: INDArray, theta: INDArray, alpha: Float, iters: Int): INDArray ={
      val temp = Nd4j.zerosLike(theta)
      val params = theta.length()
      val nbOfTrainingExamples = x.rows
      val thetas = (0 to iters).foldLeft(temp)({
         case (accum, i) =>
            val error = x.mmul(accum.T) - y
            (0 until params).map{
               p =>
                  val r2 =  accum.getFloat(0, p) - (error * x.getColumn(p)).sum(0).mul(alpha/nbOfTrainingExamples).getFloat(0)
                  accum.put(0, p, r2)
            }
            println(s"Cost: ${computeCost(x, y, accum)}")
            accum
      })
      thetas
   }

}



