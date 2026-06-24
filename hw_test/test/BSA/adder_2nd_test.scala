package ctp

import org.scalatest._
import chiseltest._
import chisel3._
import chisel3.util._
import scala.util.Random

class Adder2_Test extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "Adder_2nd"
  it should "produce right output" in {
    //Add your own functions here
    //Add your own values here
    test(new Adder_2nd(32)) { c =>
      // Prepare Data
      val iter = 10
      val in0 = List.fill(iter)(Random.nextInt(8))//(1 to iter).toList//for (i <- 1 to iter) yield i
      val in1 = List.fill(iter)(Random.nextInt(8))//(1 to iter).toList//for (i <- 1 to iter) yield i
      var acc = 0//List.fill(iter)(Random.nextInt(8))//for (i <- 1 to iter) yield Random.nextInt(2)

      // 초기 리스트를 0으로 채움 (C0, C1)
      var answer0 = (in0 ++ in1).sum
      var answer1 = List.fill(iter)(0)

      // Runtime
      for (cycle <- 0 until iter) {
        c.io.in0.poke(in0(cycle).U)
        c.io.in1.poke(in1(cycle).U)
        c.io.in2.poke(acc.U)

        c.clock.step(1)
        acc = c.io.out.peek().litValue.toInt 

        println(s">> Cycle $cycle: ins = [${in0(cycle)}] & [${in1(cycle)}] : outs = [$acc]")
        
      }
      println(s"\n>> Answer : ${answer0}")
      //println(s"Iter[2] End, have to be $answer2\n")
    }
  }
}
