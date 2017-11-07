package com.cpuheater.ml.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class GridWorld(size: Int, terminateStates: List[Int], default: Int = -1, terminalDefault: Int=0) {
  import GridWorld._


  private val grid = (0 until size*size).map{
    index =>
    if(terminateStates.exists(_ == index))
      terminalDefault
    else
      default
  }

  def getSize = size


  def getStates : List[Int] = {
    (0 until grid.length).toList
  }


  def getActions : List[Int] = List(UP, RIGHT, DOWN,  LEFT)

  private def isTerminal(pos: Int): Boolean =
    terminateStates.exists(pos == _)


  private def toRowCol(pos: Int): (Int, Int) =  {
   val row = (pos)/ size
   val col = (pos) % size
   (row, col)
  }

  private def beyondBorders(row: Int, col: Int): Boolean = {
    (row, col) match {
      case (row, col) if (row < 0 || col < 0 ) || (row >= size || col >= size)  =>
         true
      case _ =>
        false
    }
  }

  private def calcPos(currentPos: Int, action: Int) = {
    val (row, col) = toRowCol(currentPos)
    action match {
      case GridWorld.UP =>
        (row+1, col)
      case GridWorld.RIGHT =>
        (row, col +1)
      case GridWorld.LEFT =>
        (row, col -1)
      case GridWorld.DOWN =>
        (row-1, col)
      case ups =>
        throw new UnsupportedOperationException(s"Unknown action ${ups}")
    }
  }

  def move(currentPos: Int,  action: Int) : (Float, Int, Float, Boolean)= {
    val (row, col) = calcPos(currentPos, action)
    beyondBorders(row, col) match {
      case true if isTerminal(currentPos)=>
        (1, currentPos, terminalDefault, true)
      case false if  isTerminal(currentPos) =>
        (1, currentPos, terminalDefault, true)
      case false if  isTerminal(row*size+col)=>
        (1, row*size+col, default, true)
      case true =>
        (1, currentPos, default, false)

      case false =>
        (1, row*size+col, default, false)
    }

  }

}



object GridWorld {

  val UP = 0
  val RIGHT = 1
  val DOWN = 2
  val LEFT = 3


 def apply(size: Int = 4, terminal: List[Int] = List(0, 15), default: Int = -1, terminalDefault: Int=0) ={

   new GridWorld(size, terminal, default, terminalDefault)

 }

 def prettyPrint[T](data: List[T], size: Int) = {
   (0 until size).foreach{
     i =>
       println(s"${data.slice(i*size, i*size +size).mkString(",")}")
   }
 }
}




