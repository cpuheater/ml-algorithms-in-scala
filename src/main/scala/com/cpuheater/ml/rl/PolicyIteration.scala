package com.cpuheater.ml.rl

import com.cpuheater.ml.util.GridWorld
import scala.collection.mutable.ListBuffer


object PolicyIteration  {

  def computePolicyIteration(gridWorld: GridWorld, gamma: Float, eps: Float): (List[List[Float]], List[Float]) = {

    val policy = ListBuffer
      .fill[List[Float]](gridWorld.getStates.length)(
      List.fill(gridWorld.getActions.length)(1f/gridWorld.getActions.length))

    var valueFunction = PolicyEvaluation.computePolicyEvaluation(gridWorld, policy.toList, gamma, eps)
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
        valueFunction = PolicyEvaluation.computePolicyEvaluation(gridWorld, policy.toList, gamma, eps)

    }
    (policy.toList, valueFunction)
  }
}