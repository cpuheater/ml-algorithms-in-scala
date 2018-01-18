package com.cpuheater.ml.rl
import com.cpuheater.ml.util.GridWorld


object PolicyEvaluation {

  def computePolicyEvaluation(gridWorld: GridWorld,
                              policy: List[List[Float]], gamma: Float, eps: Float): List[Float] = {
    val states = gridWorld.getStates
    val actions = gridWorld.getActions
    var continue = true
    val valueFunction = scala.collection.mutable.ListBuffer.fill(gridWorld.getAllStates)(0.0f)

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
