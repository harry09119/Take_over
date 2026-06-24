package colcomb

import chisel3._
import chisel3.util._
import scala.math._
import chisel3.util.log2Ceil


/* Column Combine Architecture */
class FA () extends Module {
  val io = IO(new Bundle {
    val a   = Input(Bool())  // 첫 번째 입력 비트
    val b   = Input(Bool())  // 두 번째 입력 비트
    val ci  = Input(Bool())  // 입력 Carry
    val s   = Output(Bool()) // 결과 합 (sum)
    val co  = Output(Bool()) // 출력 Carry (carry-out)
  })

  io.s  := io.a ^ io.b ^ io.ci
  io.co := (io.a & io.b) | (io.b & io.ci) | (io.a & io.ci)
}

class BsMAC(width: Int) extends Module {
  val io = IO(new Bundle {
    //signals
    val reset   = Input(Bool())

    //operands
    val Xi      = Input(Bool())         // Cycle마다 1비트 입력
    val W       = Input(UInt(width.W))  // 새로운 Weight 입력

    val Yi      = Input(Bool()) // Partial Sum 입력 (누적)
    val Yo      = Output(Bool()) // 누적된 Partial Sum 출력

    val debug   = Output(Bool())
  })

  val FAs = Seq.fill(width+1)(Module(new FA()))  
  val BCo = Seq.fill(width+1)(RegInit(false.B))  
  val SCi = Seq.fill(width+1)(RegInit(false.B))

  for(i <- 0 until width) {
    FAs(i).io.a := io.W(width - i - 1) & io.Xi
    FAs(i).io.b := BCo(i)//RegNext(Mux(FAs(i).io.co), 0.B)
    if (i > 0)
      FAs(i).io.ci := SCi(i-1)//RegNext(FAs(i-1).io.s)
    
    when(io.reset){
      SCi(i) := 0.B
      BCo(i) := 0.B
    }.otherwise {
      SCi(i) := FAs(i).io.s
      BCo(i) := FAs(i).io.co
    }
  }
  
  io.debug := SCi(width-1)
  FAs(0).io.ci := 0.B
  
  FAs(width).io.a := SCi(width-1)//RegNext(FAs(width-1).io.s, 0.U)//SCi(width-1)
  FAs(width).io.b := BCo(width)//RegNext(FAs(width).io.co, 0.U)//BCo(width)
  FAs(width).io.ci := io.Yi

  BCo(width) := FAs(width).io.co
  SCi(width) := FAs(width).io.s
  io.Yo := SCi(width)//RegNext(FAs(width).io.s, 0.U)//SCi(width)
}


class BsPE(width: Int, parallel: Int) extends Module {
  val tag_width = log2Ceil(parallel)
  val io = IO(new Bundle {
    //signals
    val reset       = Input(Bool())
    val turn        = Input(UInt(2.W))
    val switch      = Input(Vec(4, Bool()))

    //operands
    val inA_bit     = Input(Vec(parallel, Bool()))         // Cycle마다 1비트 입력

    val inB         = Input(Vec(4, UInt(width.W)))  // 새로운 Weight 입력
    val outB        = Output(Vec(4, UInt(width.W)))
    
    val inT         = Input(Vec(4,UInt(tag_width.W)))
    val outT        = Output(Vec(4, UInt(tag_width.W)))

    val inC_bit     = Input(Vec(4, Bool())) // Partial Sum 입력 (누적)
    val outC_bit    = Output(Vec(4, Bool())) // 누적된 Partial Sum 출력

    val debug       = Output(Vec(4, Bool()))
  })
  
  val macs = Seq.fill(4)(Module(new BsMAC(width)))

  val wgt0 = Reg(Vec(4, UInt(width.W)))
  val wgt1 = Reg(Vec(4, UInt(width.W)))

  val tag0 = Reg(Vec(4, UInt(tag_width.W)))
  val tag1 = Reg(Vec(4, UInt(tag_width.W)))

  for (i <- 0 until 4) {
    //connection
    macs(i).io.reset  := io.reset
    macs(i).io.Yi     := io.inC_bit(i)
    io.outC_bit(i)    := macs(i).io.Yo 
    
    when(io.reset) {
      macs(i).io.Xi := 0.U
      macs(i).io.W  := 0.U
      io.outB(i)  := 0.U
      io.outT(i)  := 0.U
      io.debug(i) := 0.U
      wgt0(i) := 0.U
      tag0(i) := 0.U
      wgt1(i) := 0.U
      tag1(i) := 0.U
    }.otherwise {
      val turn = (io.turn === i.U)
      
      val chosen_inA  = Mux(io.switch(i), io.inA_bit(tag0(i)), io.inA_bit(tag1(i)))
      val compute_wgt = Mux(io.switch(i), wgt0(i), wgt1(i))
      macs(i).io.Xi := Mux(turn, chosen_inA ,0.U)
      io.debug(i)   := Mux(turn, chosen_inA ,0.U)

      macs(i).io.W  := compute_wgt

      val load_wgt    = Mux(io.switch(i), wgt1(i), wgt0(i))
      val load_tag    = Mux(io.switch(i), tag1(i), tag0(i))
      io.outB(i)  := load_wgt
      io.outT(i)  := load_tag

      when(io.switch(i)) {
        wgt1(i)     := io.inB(i)
        tag1(i)     := io.inT(i)
      }.otherwise {
        wgt0(i)     := io.inB(i)
        tag0(i)     := io.inT(i)
      }
    }
  }
}

class BsPErow(width: Int, parallel: Int, rowL: Int) extends Module {
  val tag_width = log2Ceil(parallel)
  val io = IO(new Bundle {
    //signals
    val reset       = Input(Bool())
    val turn        = Input(Vec(rowL, UInt(2.W)))
    val switch      = Input(Vec(rowL, Vec(4, Bool())))

    //operands
    val inA_bit     = Input(Vec(rowL, Vec(parallel, Bool())))         // Cycle마다 1비트 입력

    val inB         = Input(Vec(4, UInt(width.W)))  // 새로운 Weight 입력

    val inT         = Input(Vec(4,UInt(tag_width.W)))

    val inC_bit     = Input(Vec(4, Bool())) // Partial Sum 입력 (누적)
    val outC_bit    = Output(Vec(4, Bool())) // 누적된 Partial Sum 출력
  })
  
  val pes = Seq.fill(rowL)(Module(new BsPE(width, parallel)))

  for (i <- 0 until rowL) {
    pes(i).io.reset := io.reset
    pes(i).io.inA_bit := io.inA_bit(i)
    pes(i).io.switch := io.switch(i)
    pes(i).io.turn  := io.turn(i)

    if(i > 0) {
      pes(i).io.inB := pes(i-1).io.outB
      pes(i).io.inT := pes(i-1).io.outT
      pes(i).io.inC_bit := pes(i-1).io.outC_bit
    }
    else {
      pes(i).io.inB := io.inB
      pes(i).io.inT := io.inT
      pes(i).io.inC_bit := io.inC_bit
    }
  }
  
  io.outC_bit := pes(rowL-1).io.outC_bit
}

class BsTile(width: Int, parallel: Int, rowL: Int, colL: Int) extends Module {
  val tag_width = log2Ceil(parallel)
  val io = IO(new Bundle {
    //signals
    val reset       = Input(Bool())
    val turn        = Input(Vec(rowL, UInt(2.W)))
    val switch      = Input(Vec(rowL, Vec(4, Bool())))

    //operands
    val inA_bit     = Input(Vec(rowL, Vec(parallel, Bool())))         // Cycle마다 1비트 입력

    val inB         = Input(Vec(colL, Vec(4, UInt(width.W))))  // 새로운 Weight 입력

    val inT         = Input(Vec(colL, Vec(4,UInt(tag_width.W))))

    val inC_bit     = Input(Vec(colL, Vec(4, Bool())))  // Partial Sum 입력 (누적)
    val outC_bit    = Output(Vec(colL, Vec(4, Bool()))) // 누적된 Partial Sum 출력
  })

  val perows = Seq.fill(colL)(Module(new BsPErow(width, parallel, rowL)))

  for (i <- 0 until colL) {
    perows(i).io.reset := io.reset
    perows(i).io.inA_bit := io.inA_bit
    perows(i).io.switch := io.switch
    perows(i).io.turn  := io.turn
    
    perows(i).io.inB := io.inB(i)
    perows(i).io.inT := io.inT(i)
    perows(i).io.inC_bit := io.inC_bit(i)

    io.outC_bit(i) := perows(i).io.outC_bit
  }
}

object BsTile {
  def apply(width:Int=8, parallel: Int=4, rowL: Int=8, colL: Int=8): BsTile = {
    new BsTile(width, parallel,rowL, colL)
  }
}
// Top-Level 모듈
object ColumnCombineHW extends App {
  val rowL = 16
  val colL = 16
  val parallel = 4
  val width = 8
  val targetDir = "/home/harry09119/chisel/src/main/scala/SA/verilog"
  val top = "Tile"
    
  if (top == "Tile") {
    val verilogFileName = s"BsTile_${rowL}_${colL}_${parallel}.v"  // 동적 파일명 생성
    println("\nGenerating Verilog of...",verilogFileName,"\n")
    (new chisel3.stage.ChiselStage).emitVerilog(
      BsTile(rowL=rowL,colL=colL,parallel = parallel),
      Array("--target-dir", targetDir, "--output-file", verilogFileName)
    )
  }

  else
    println("\nLook carefully")
}

