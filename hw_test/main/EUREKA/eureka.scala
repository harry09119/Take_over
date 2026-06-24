package eureka

import chisel3._
import chisel3.util._
import scala.math._
import chisel3.util.log2Ceil

class Adder(width: Int) extends Module {
  val io = IO(new Bundle {
    val in0   = Input(UInt(width.W))
    val in1   = Input(UInt(width.W))
    val in2   = Input(UInt(width.W))
    val out   = Output(UInt(width.W))
  })

  val sum = io.in0 ^ io.in1 ^ io.in2
  val carry = (io.in0 & io.in1) | (io.in0 & io.in2) | (io.in1 & io.in2)
  val result = (sum + (carry << 1))(31,0)

  io.out := result
}

class SpPE(val width: Int, val parallel:Int) extends Module {
  def log2(x: Int): Int = log2Ceil(x)
  val io = IO(new Bundle {
    val inA = Input(Vec(parallel, UInt(width.W)))   // 입력 입력값
    val inB = Input(UInt(width.W))   // 입력 가중치 
    val tag = Input(UInt(log2(parallel).W))
    val upC = Output(UInt(32.W))
    val downC = Input(UInt(32.W))
    val reset = Input(Bool())     // register reset
    val totop = Input(Bool())
  })
  // 두 개의 출력 레지스터
  val out = RegInit(0.U(32.W))

  // Multiply and Accumulate
  val mul = io.inA(io.tag) * io.inB
  val updowns = Mux(io.totop, io.downC, io.downC + mul)

  out := Mux(io.reset, 0.U, updowns + out)

  io.upC := Mux(io.totop, mul, 0.U)
}

object SpPE {
  def apply(width:Int=8, parallel: Int=4): SpPE = {
    new SpPE(width, parallel)
  }
}

class SpPErow(val parallel:Int, val rowL: Int, val width: Int) extends Module {
  def log2(x: Int): Int = log2Ceil(x)
  val io = IO(new Bundle {
    val inA   = Input(Vec(rowL, Vec(parallel, UInt(width.W))))  // "rowL"개의 입력값
    val inB   = Input(UInt(width.W))  // 1개의 가중치
    val tag   = Input(UInt(log2(parallel).W))
  
    val upC   = Output(Vec(rowL,UInt(32.W)))
    val downC = Input(Vec(rowL,UInt(32.W)))
    val totop = Input(Bool())

    val reset = Input(Bool())     // register reset
  })

  // 2D Array of Processing Elements
  val perow = Seq.fill(rowL)(Module(SpPE(width=width, parallel = parallel)))

  // Connect inputs to PEs and outputs

  for (i <- 0 until rowL) {
    for (j <- 0 until parallel)
      perow(i).io.inA(j) := io.inA(i)(j)  // 행 단위로 inA 연결
    perow(i).io.inB := io.inB       // 열 단위로 inB 연결
    perow(i).io.tag := io.tag
    io.upC(i) := perow(i).io.upC
    perow(i).io.downC := io.downC(i)
    perow(i).io.totop := io.totop

    perow(i).io.reset := io.reset
  }
}

object SpPErow {
  def apply(width:Int=8, parallel:Int=4, rowL:Int=8): SpPErow = {
    new SpPErow(parallel, rowL, width)
  }
}

class SpTile(val rowL: Int, val colL: Int, val width: Int, val parallel: Int) extends Module {
  val io = IO(new Bundle {
    val inA = Input(Vec(rowL, Vec(parallel, UInt(width.W))))  // "rowL"개의 입력값
    val inB = Input(Vec(colL, UInt(width.W)))  // 1개의 가중치

    val outA = Output(Vec(rowL, Vec(parallel, UInt(width.W))))  // "rowL"개의 입력값
    val outB = Output(Vec(colL, UInt(width.W)))  // 1개의 가중치
 
    val tag = Input(Vec(colL, UInt(log2Ceil(parallel).W)))
    val reset = Input(Bool())     // register reset
    val totop = Input(Vec(colL, Bool()))
  })

  // 2D Array of Processing Elements
  val tile = Seq.fill(colL)(Module(SpPErow(rowL=rowL,parallel=parallel, width=width)))

  val a_regs = RegInit(
    VecInit(Seq.fill(rowL)(
        VecInit(Seq.fill(parallel)(0.U(width.W)))
    ))
  )

  val b_regs = RegInit(VecInit(Seq.fill(colL)(0.U(width.W))))

  val a_zeros = VecInit(Seq.fill(rowL)(
        VecInit(Seq.fill(parallel)(0.U(width.W)))
      ))

  val b_zeros = VecInit(Seq.fill(colL)(0.U(width.W)))

  for(i <- 0 until rowL) {
    a_regs := Mux(io.reset, a_zeros, io.inA)
    io.outA := a_regs
  }

  for(i <- 0 until colL) {
    b_regs := Mux(io.reset, b_zeros, io.inB)
    io.outB := b_regs
  }

  for (c <- 0 until colL) {
    for (i <- 0 until rowL) {
      for (j <- 0 until parallel)
        tile(c).io.inA(i)(j) := io.inA(i)(j)  // 행 단위로 inA 연결
    }
  }
  
  for (i <- 0 until colL){
    tile(i).io.inB := io.inB(i)       // 열 단위로 inB 연결
    tile(i).io.tag := io.tag(i)
    tile(i).io.reset := io.reset
    tile(i).io.totop := io.totop(i)
  
    if(i > 0)
      tile(i).io.downC := tile(i-1).io.upC
    else
      tile(i).io.downC := VecInit(Seq.fill(rowL)(0.U(32.W)))
  }
}

object SpTile {
  def apply(width:Int=8, parallel:Int=4, rowL:Int=8,colL:Int=8): SpTile = {
    new SpTile(rowL,colL,width,parallel)
  }
}

object Eureka_new_To_Verilog extends App {
  val rowL = 16
  val colL = 16
  val parallel = 16
  val targetDir = "/home/harry09119/chisel/src/main/scala/SA/verilog"
  val top = "Tile"
    
  if (top == "Tile") {
    val verilogFileName = s"Eureka_new_Tile_${rowL}_${colL}_${parallel}.v"  // 동적 파일명 생성
    println("\nGenerating Verilog of...",verilogFileName,"\n")
    (new chisel3.stage.ChiselStage).emitVerilog(
      SpTile(rowL=rowL,colL=colL,parallel = parallel),
      Array("--target-dir", targetDir, "--output-file", verilogFileName)
    )
  }
}
