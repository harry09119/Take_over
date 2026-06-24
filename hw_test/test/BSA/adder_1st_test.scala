package ctp

import org.scalatest._
import chiseltest._
import chisel3._
import chisel3.util._
import scala.util.Random

class Adder1_Test extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "Adder_1st"
  it should "produce right output" in {
    //Add your own functions here
    //Add your own values here
    test(new Adder_1st(16, 0)) { c =>
      // Prepare Data
      val iter = 10
      val in0 = List.fill(iter)(Random.nextInt(8))//(1 to iter).toList//for (i <- 1 to iter) yield i
      val in1 = List.fill(iter)(Random.nextInt(8))//(1 to iter).toList//for (i <- 1 to iter) yield i
      val s0 = List.fill(iter)(Random.nextInt(2))//for (i <- 1 to iter) yield Random.nextInt(2)
      val s1 = List.fill(iter)(Random.nextInt(2))//for (i <- 1 to iter) yield Random.nextInt(2)

      // 초기 리스트를 0으로 채움 (C0, C1)
      var answer0 = List.fill(iter)(0)
      var answer1 = List.fill(iter)(0)

      // 조건에 따라 C0, C1 값 설정
      answer0 = in0.zip(in1).zip(s0.zip(s1)).map {
        case ((a, b), (sa, sb)) => 
          if (sa == sb && sa == 0) a + b // sA[i] == sB[i] == 0 -> C0[i] = A[i] + B[i]
          else if (sa != sb && sa == 0) a
          else if (sa != sb && sa == 1) b
          else 0
      }

      answer1 =  in0.zip(in1).zip(s0.zip(s1)).map {
        case ((a, b), (sa, sb)) =>
          if (sa == sb && sa == 1) a + b // sA[i] == sB[i] == 0 -> C0[i] = A[i] + B[i]
          else if (sa != sb && sa == 0) b
          else if (sa != sb && sa == 1) a
          else 0
      }

      // Set Up
      c.io.reset.poke(true.B)
      c.clock.step(1) // reset 활성화

      c.io.reset.poke(false.B) // reset 비활성화
      // Runtime
      for (cycle <- 0 until iter) {
        c.io.ins(0).poke(in0(cycle).U)
        c.io.ins(1).poke(in1(cycle).U)
        c.io.ss(0).poke(s0(cycle).U)
        c.io.ss(1).poke(s1(cycle).U) 

        c.clock.step(1)

        val out0 = c.io.out(0).peek().litValue.toInt
        val out1 = c.io.out(1).peek().litValue.toInt

        println(s">> Cycle $cycle: ins = [${in0(cycle)}, ${s0(cycle)}] & [${in1(cycle)}, ${s1(cycle)}] : outs = [$out0, $out1]")
        println(s">>>> Answer ${cycle}: ${answer0(cycle)}, ${answer1(cycle)}")
        
      }
      //println(s"Iter[2] End, have to be $answer2\n")
    }
  }
}
