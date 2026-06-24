package ctp

import org.scalatest._
import chiseltest._
import chisel3._
import chisel3.util._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class PEStack_Test extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "PEStack"
  it should "produce right output" in {
    //Add your own functions here
    //Add your own values here
    val rowL = 4
    val b_width = 8
    val parallel = 4
    val t_width = log2Ceil(parallel)
    test(new PEStack(b_width, parallel)) { c =>
      // Prepare Data
      val epoch = 3
      val length = 32
      val duration = 8

      val test_W = List.fill(length)(Random.nextInt(16))
      val test_T = List.fill(length)(Random.nextInt(parallel))
      val test_S = List.fill(length)(Random.nextInt(2))
      val test_I = List.fill(length, parallel)(Random.nextInt(32))
      
      var multiply = (test_I zip (test_W zip test_T)).map { case (a, (b, c)) => a(c) * b }
      var answer0  = (multiply zip test_S).map{ case (a, b) => if(b==0) a else 0}.sum
      var answer1  = (multiply zip test_S).map{ case (a, b) => if(b==1) a else 0}.sum
      
      // Set

      c.io.ctrl.poke(0.U)
      c.clock.step(1) 
      
      c.io.ctrl.poke(3.U)
      c.io.inC.poke(0.U)

      // Runtime
      
      var w_data = List.fill(4)(0)
      var t_data = List.fill(4)(0)
      var s_data = List.fill(4)(0)
      var i_data = List.fill(4,parallel)(0)
      var out_data = Array.fill(2)(-1)

      for (cycle <- 0 until duration + 1) {
        
        if(cycle < duration) {
          val pointer = cycle*4
          w_data = test_W.slice(pointer,pointer+4)
          t_data = test_T.slice(pointer,pointer+4)
          s_data = test_S.slice(pointer,pointer+4)
          i_data = test_I.slice(pointer,pointer+4)
        }

        else {
          w_data = List.fill(4)(0)
          t_data = List.fill(4)(0)
          s_data = List.fill(4)(0)
          i_data = List.fill(4, parallel)(0)  
        }

        for (i <- 0 until 4) {
          c.io.inB(i).poke(w_data(i).U)
          c.io.inT(i).poke(t_data(i).U)
          c.io.inS(i).poke(s_data(i).U)

          for (j <- 0 until parallel) {
            c.io.inA(i)(j).poke(i_data(i)(j).U)
          }
        }

        c.clock.step(1) // 한 사이클 진행 후 출력 확인

        out_data(0) = c.io.outC(0).peek().litValue.toInt
        out_data(1) = c.io.outC(1).peek().litValue.toInt

        val sel_i_data = (i_data zip t_data).map{case (a, b) => a(b)}

        println(s"> Cycle $cycle: In = [$w_data|$sel_i_data] : out = [${out_data(0)},${out_data(1)}]")
      }

      println(s">>> Answer: [$answer0, $answer1]")
    }
  }
}
