package com.cpuheater.ml.supervised

import com.cpuheater.ml.rl.{PolicyIteration, QLearning}
import com.cpuheater.ml.util.{BagOfWordsTransformer, GridWorld, TestSupport}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

class QLearningSpec  extends TestSupport{


  it should "Q learning" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

    val eps = 0.1f
    val gamma = 0.9f
    val alpha = 0.1f
    val gridWorld = GridWorld(default = -0.1)

    val qValue = QLearning.calc(gridWorld, alpha, gamma, eps)
    println("Q Value")
    GridWorld.prettyPrint(qValue, gridWorld.getSize)
    /*println("Policy")

    val optimalPolicyActions = optimalPolicy.flatMap{
      row =>
        row.zipWithIndex.find(_._1>0).map(_._2)
    }
    GridWorld.prettyPrint(optimalPolicyActions, gridWorld.getSize)*/

  }


}
