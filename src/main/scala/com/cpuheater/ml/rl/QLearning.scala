package com.cpuheater.ml.rl

import com.cpuheater.ml.util.GridWorld

import scala.collection.mutable.ListBuffer

object QLearning {


  def calc(gridWorld: GridWorld, alpha: Double, gamma: Double, eps: Double): List[Double] = {
    val states = gridWorld.getStates
    val actions = gridWorld.getActions
    var continue = true
    val qValueFun = scala.collection.mutable.ListBuffer.fill(gridWorld.getAllStates)(ListBuffer.fill(actions.length)(0.0))

    val deltas = new ListBuffer[Double]()

    (0 until 1000).foreach {
      epoch =>

        var currenState = 0
        var gameOver = false
        var currentAction = argMax(qValueFun(currenState).toList)
        var delta = 0.0
        while (!gameOver) {
          currentAction = randomAction(currentAction, actions)

          val (_, nextState, reward, done) = gridWorld.move(currenState, currentAction)
          val currentQ = qValueFun(currenState)(currentAction)

          val nextMaxAction = argMax(qValueFun(nextState).toList)
          val nextMaxQ = qValueFun(nextState).max

          qValueFun(currenState)(currentAction) = qValueFun(currenState)(currentAction) + alpha * (reward + gamma * nextMaxQ - qValueFun(currenState)(currentAction))
          delta = Math.max(delta, currentQ - qValueFun(currenState)(currentAction))
          deltas += delta
          currenState = nextState
          gameOver = done
        }

    }
    qValueFun.toList.map(_.toList.max)
  }


  private def randomAction(a: Int, allActions: List[Int], eps: Float = 0.1f): Int = {
    if(Math.random() < eps) {
      a
    }
    else {
      val r = scala.util.Random.nextInt(allActions.length)
      allActions(r)
    }
  }

  private def argMax(list: List[Double]): Int = list.zipWithIndex.maxBy(_._1)._2


}
