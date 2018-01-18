package com.cpuheater.ml.supervised

import com.cpuheater.ml.rl.PolicyEvaluation.computePolicyEvaluation
import com.cpuheater.ml.util.{BagOfWordsTransformer, GridWorld, TestSupport}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

class PolicyEvaluationSpec  extends TestSupport{


  it should "policy evaluation" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

    val gridWorld = GridWorld()
    val eps = 0.00001f
    val gamma = 1

    val policy = List
      .fill[List[Float]](gridWorld.getAllStates)(
      List.fill(gridWorld.getActions.length)(1f/gridWorld.getActions.length))

    val valueFunction = computePolicyEvaluation(gridWorld, policy, gamma, eps)

    GridWorld.prettyPrint(valueFunction, gridWorld.getSize)
  }


}
