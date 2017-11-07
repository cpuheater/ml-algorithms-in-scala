package com.cpuheater.ml.rl
import com.cpuheater.ml.rl.PolicyEvaluation.gridWorld
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import com.cpuheater.ml.util.GridWorld
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

object PolicyEvaluation extends App with PolicyEvaluationUtil{
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)

  val gridWorld = GridWorld()
  val eps = 0.00001f
  val gamma = 1

  val policy = List
    .fill[List[Float]](gridWorld.getStates.length)(
    List.fill(gridWorld.getActions.length)(1f/gridWorld.getActions.length))

  val valueFunction = computePolicyEvaluation(gridWorld, policy, gamma, eps)

  GridWorld.prettyPrint(valueFunction, gridWorld.getSize)

}


trait PolicyEvaluationUtil {

  def computePolicyEvaluation(gridWorld: GridWorld, policy: List[List[Float]], gamma: Float, eps: Float)
  : List[Float] = {
    val states = gridWorld.getStates
    val actions = gridWorld.getActions
    var continue = true
    val valueFunction = scala.collection.mutable.ListBuffer.fill(states.length)(0.0f)

    var epoch = 0
    while(continue) {
      epoch = epoch +1
      var delta = 0.0f
      states.foreach {
        state =>
          val actionProbs = policy(state)
          var value = 0.0f
          actionProbs.zip(actions).foreach {
            case (actionProb, action) =>
              val (prob, nextState, reward, done) = gridWorld.move(state, action)
              value = value + actionProb * prob * (reward + gamma*valueFunction(nextState))
          }
          delta = Math.max(delta, Math.abs(value - valueFunction(state)))
          valueFunction(state) = value
      }
      if(delta <= eps){
        continue = false
      }
    }
    valueFunction.toList
  }

}
