package com.cpuheater.ml.rl
import com.cpuheater.ml.rl.PolicyEvaluation.gridWorld
import com.cpuheater.ml.rl.PolicyIteration.{computePolicyEvaluation, eps, gamma}
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import com.cpuheater.ml.util.GridWorld
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

import scala.collection.mutable.ListBuffer

object PolicyIteration extends App with PolicyIterationUtil {
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT)


  val eps = 0.00001f
  val gamma = 1
  val gridWorld = GridWorld()

  val (optimalPolicy, valueFunction) = computePolicyIteration(gridWorld)
  println("Value Function")
  GridWorld.prettyPrint(valueFunction, gridWorld.getSize)
  println("Policy")

  val optimalPolicyActions = optimalPolicy.flatMap{
    row =>
      row.zipWithIndex.find(_._1>0).map(_._2)
  }
  GridWorld.prettyPrint(optimalPolicyActions, gridWorld.getSize)


}


trait PolicyIterationUtil extends PolicyEvaluationUtil {




  def computePolicyIteration(gridWorld: GridWorld): (List[List[Float]], List[Float]) = {

    val policy = ListBuffer
      .fill[List[Float]](gridWorld.getStates.length)(
      List.fill(gridWorld.getActions.length)(1f/gridWorld.getActions.length))

    var valueFunction = computePolicyEvaluation(gridWorld, policy.toList, gamma, eps)
    var continue = true
    while(continue) {
      val states = gridWorld.getStates

      var policyConverged = true

      states.map{
        state =>
          val currentMaxAction = policy(state).zipWithIndex.maxBy(_._1)._2
          val actionValues = scala.collection.mutable.ListBuffer.fill(gridWorld.getActions.length)(0.0f)
          (0 until gridWorld.getActions.length).map{
            action =>
              val (prob, nextState, reward, done) = gridWorld.move(state, action)
              actionValues(action) = actionValues(action) + prob *(reward + gamma * valueFunction(nextState))
          }
          val maxAction = actionValues.zipWithIndex.maxBy(_._1)._2

          if(maxAction != currentMaxAction){
            policyConverged = false
          }
          policy(state) = List.fill(gridWorld.getActions.length)(0f).updated(maxAction, 1f)
      }

      if(policyConverged)
        continue = false
      else
        valueFunction = computePolicyEvaluation(gridWorld, policy.toList, gamma, eps)

    }
    (policy.toList, valueFunction)
  }
}