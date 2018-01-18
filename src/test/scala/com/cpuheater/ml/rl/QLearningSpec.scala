package com.cpuheater.ml.supervised

import com.cpuheater.ml.rl.PolicyIteration
import com.cpuheater.ml.util.{BagOfWordsTransformer, GridWorld, TestSupport}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

class QLearningSpec  extends TestSupport{


  it should "policy iteration" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

    val eps = 0.00001f
    val gamma = 1
    val gridWorld = GridWorld()

    val (optimalPolicy, valueFunction) = PolicyIteration.computePolicyIteration(gridWorld, gamma, eps)
    println("Value Function")
    GridWorld.prettyPrint(valueFunction, gridWorld.getSize)
    println("Policy")

    val optimalPolicyActions = optimalPolicy.flatMap{
      row =>
        row.zipWithIndex.find(_._1>0).map(_._2)
    }
    GridWorld.prettyPrint(optimalPolicyActions, gridWorld.getSize)

  }


}
