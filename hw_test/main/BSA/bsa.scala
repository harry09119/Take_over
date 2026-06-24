package ctp

import chisel3._
import chisel3.util._
import scala.math._
import chisel3.util.log2Ceil

class Buffer(depth:Int, width:Int, b_width:Int) extends Module {
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write  = Input(Bool())
    val addr   = Input(UInt(32.W))
    val wdata  = Input(Vec(width, UInt(b_width.W)))
    val rdata  = Output(Vec(width, UInt(b_width.W)))
  })

  val mem = SyncReadMem(depth, Vec(width, UInt(b_width.W)))
  
  io.rdata := DontCare
  
  when(io.enable) {
    val slot = mem(io.addr)
    when (io.write) { 
      slot := io.wdata 
    }.otherwise { 
      io.rdata := slot
    }
  }
}

class Adder_2nd(width: Int) extends Module {
  val io = IO(new Bundle {
    val in0   = Input(UInt(width.W))
    val in1   = Input(UInt(width.W))
    val in2   = Input(UInt(width.W))
    // carry save adder는 두 개의 출력(sum과 carry)을 생성합니다.
    val out   = Output(UInt(width.W))
  })
  
  //val sum = io.in0 ^ io.in1 ^ io.in2
  //val carry = (io.in0 & io.in1) | (io.in0 & io.in2) | (io.in1 & io.in2)
  //val result = (sum + (carry << 1))(31,0)

  //io.out := result
  io.out := io.in0 + io.in1 + io.in2
}

class Adder_1st(width: Int, id: Int) extends Module {
  // b_width를 width로 사용 (예: b_width = 16)
  val io = IO(new Bundle {
    val ins   = Input(Vec(2, UInt(width.W)))
    val ss    = Input(Vec(2, UInt(1.W)))
    val out   = Output(Vec(2, UInt(32.W)))
    val reset = Input(Bool())
  })

  // -------------------------------------
  // 1. 조건 신호 생성 (조합 논리)
  // 입력 IO.ss를 기반으로 to_adder 신호 생성:
  val to_adder = ~(io.ss(0) ^ io.ss(1))

  // -------------------------------------
  // 2. 내부 상태 레지스터 (최소화)
  // 내부 ss 레지스터를 regSS라 명명 (2비트)
  val regSS  = RegInit(0.U(2.W))
  // 내부 결과 레지스터 (32비트)
  val regOut = RegInit(0.U(32.W))

  regSS := Mux(io.reset, 0.U(2.W), Cat(io.ss(1), io.ss(0)))
  
  val out_data = Mux(to_adder === 1.U,io.ins(0) + io.ins(1),Cat(io.ins(0)(width-1, 0), io.ins(1)(width-1, 0)))
  regOut := Mux( io.reset, 0.U(32.W), out_data)

  val is_sum     = ~(regSS(0) ^ regSS(1))
  val to_forward = regSS(0) & regSS(1)

  io.out(0) := Mux(is_sum === 1.U,
                   Mux(to_forward === id.U, regOut, 0.U),
                   Mux(regSS(0) === id.U, regOut(31, 16), regOut(15, 0))
                  )
  io.out(1) := Mux(is_sum === 1.U,
                   Mux(to_forward === id.U, 0.U, regOut),
                   Mux(regSS(0) === id.U, regOut(15, 0), regOut(31, 16))
                  )
}

class PEStack(val b_width: Int, val parallel:Int) extends Module {
  def log2(x: Int): Int = log2Ceil(x)
  def t_width = log2(parallel)
  def s_width = 1//log2(p_width/2)
  val io = IO(new Bundle {
    val inA = Input(Vec(4, Vec(parallel, UInt(b_width.W))))   // 입력 입력값
    val inB = Input(Vec(4, UInt(b_width.W)))   // 입력 가중치 
    val inC = Input(UInt(32.W))   // 입력 합산 값

    val inT = Input(Vec(4, UInt(t_width.W)))
    val inS = Input(Vec(4, UInt(s_width.W)))

    val outC = Output(UInt(32.W)) // 다음 PE로 전달될 완성된 출력값
    val ctrl = Input(UInt(2.W))
    val reset = Input(UInt(1.W))
    //tag bits
    //00->10->11->01->00
    //0   2   3   1   0
    
    //output reg usage along tile progress
    //11->m1->0m->10->11
    //0m->10->11->m1->0m
  })
  
  val outs = Reg(Vec(4, UInt(32.W)))

  // Multiply
  val muls = Wire(Vec(4, UInt(16.W)))
  for(i <- 0 until 4) {
    muls(i) := io.inA(i)(io.inT(i)) * io.inB(i)
  }
  
  val is_reset = (io.reset === 1.U)

  val left_adder_1st = Module(new Adder_1st(16, 0))
  val left_adder_2nd = Module(new Adder_2nd(32))
  //val as_left_out = Mux(io.ctrl1(0), left_adder_2nd.io.out, io.inC)

  left_adder_1st.io.reset  := is_reset
  left_adder_1st.io.ins(0) := muls(0)
  left_adder_1st.io.ins(1) := muls(1)

  left_adder_1st.io.ss(0) := io.inS(0)
  left_adder_1st.io.ss(1) := io.inS(1)

  left_adder_2nd.io.in0 := left_adder_1st.io.out(0)
  
  left_adder_2nd.io.in2 := Mux(io.ctrl(1),outs(2),outs(0))

  outs(0) := Mux(is_reset, 0.U(32.W), Mux(io.ctrl(1),io.inC,left_adder_2nd.io.out))
  outs(2) := Mux(is_reset, 0.U(32.W), Mux(io.ctrl(1),left_adder_2nd.io.out,io.inC))

  val right_adder_1st = Module(new Adder_1st(16, 1))
  val right_adder_2nd = Module(new Adder_2nd(32))
  //val as_right_out = Mux(io.ctrl1(1), right_adder_2nd.io.out, io.inC)
  
  right_adder_1st.io.reset  := is_reset
  right_adder_1st.io.ins(0) := muls(2)
  right_adder_1st.io.ins(1) := muls(3)

  right_adder_1st.io.ss(0) := io.inS(2)
  right_adder_1st.io.ss(1) := io.inS(3)
  
  right_adder_2nd.io.in0 := right_adder_1st.io.out(0)

  right_adder_2nd.io.in2 := Mux(io.ctrl(0),outs(3),outs(1))

  //outs(1) := Mux(is_reset, 0.U(32.W), as_right_out)
  //outs(3) := Mux(is_reset, 0.U(32.W), as_right_out)

  outs(1) := Mux(is_reset, 0.U(32.W), Mux(io.ctrl(0),io.inC,right_adder_2nd.io.out))
  outs(3) := Mux(is_reset, 0.U(32.W), Mux(io.ctrl(0),right_adder_2nd.io.out, io.inC))

  //Cross
  left_adder_2nd.io.in1 := right_adder_1st.io.out(1)
  right_adder_2nd.io.in1 := left_adder_1st.io.out(1)

  //io.outC := Mux(io.ctrl(0), outs(1), outs(0))
  io.outC := MuxLookup(io.ctrl, outs(3), Seq(
    0.U -> outs(3),
    1.U -> outs(2),
    2.U -> outs(0),
    3.U -> outs(1)
  ))

  //io.outC := outs
}

object PEStack {
  def apply(b_width:Int=8, parallel: Int=4): PEStack = {
    new PEStack(b_width, parallel)
  }
}

class MyTile(val b_width: Int, val parallel:Int, val rowL:Int, val colL:Int) extends Module {
  def log2(x: Int): Int = log2Ceil(x)
  def t_width = log2(parallel)
  def s_width = 1//log2(p_width/2)
  val io = IO(new Bundle {
    val inA = Input(Vec(rowL/4, Vec(4, Vec(parallel, UInt(b_width.W)))))   // 입력 입력값
    val inB = Input(Vec(colL, Vec(4, UInt(b_width.W))))   // 입력 가중치
    val inC = Input(Vec(colL, UInt(32.W)))   // 입력 합산 값

    val inT = Input(Vec(colL, Vec(4, UInt(t_width.W))))
    val inS = Input(Vec(colL, Vec(4, UInt(s_width.W))))

    val outC = Output(Vec(colL, UInt(32.W))) // 다음 PE로 전달될 완성된 출력값
    
    val ctrl = Input(UInt(2.W))
    val reset = Input(UInt(1.W))
  })
  
  val pes = Seq.fill(colL, rowL/4)(Module(new PEStack(b_width,parallel)))

  for(j <- 0 until rowL/4) {
    for (i <- 0 until colL) {

      pes(i)(j).io.inA := io.inA(j)
      pes(i)(j).io.inB := io.inB(i)
      pes(i)(j).io.inT := io.inT(i)
      pes(i)(j).io.inS := io.inS(i)
      pes(i)(j).io.ctrl := io.ctrl
      pes(i)(j).io.reset := io.reset
      
      if(j > 0) {
        pes(i)(j-1).io.inC := pes(i)(j).io.outC
      }
      else {
        io.outC(i) := pes(i)(0).io.outC
      }
      
      if(j == (rowL/4)-1) {
        pes(i)(j).io.inC := io.inC(i)
      }
    }
  }
}

object MyTile {
  def apply(b_width:Int=8, parallel: Int=4, rowL: Int=8, colL: Int=8): MyTile = {
    new MyTile(b_width, parallel,rowL,colL)
  }
}

/*
class PEStack_SRAM(val b_width: Int, val parallel: Int) extends Module {
  def t_width = log2Ceil(parallel)
  def s_width = 1

  // 버퍼 깊이 (예제에서는 32)
  val depth = 32

  val io = IO(new Bundle {
    // Weight AXI 인터페이스
    val wdata   = Input(Vec(4, UInt(b_width.W)))
    val waddr   = Input(UInt(6.W))
    val wvalid  = Input(Bool())
    val wlast   = Input(Bool())
    val wready  = Output(Bool())
    // Input AXI 인터페이스
    val idata   = Input(Vec(4, Vec(parallel, UInt(b_width.W))))
    val iaddr   = Input(UInt(6.W))
    val ivalid  = Input(Bool())
    val ilast   = Input(Bool())
    val iready  = Output(Bool())
    // Tag AXI 인터페이스
    val tdata   = Input(Vec(4, UInt((t_width+1).W)))
    val taddr   = Input(UInt(6.W))
    val tvalid  = Input(Bool())
    val tlast   = Input(Bool())
    val tready  = Output(Bool())
    // Output (PEStack 결과)
    val odata   = Output(Vec(2, UInt(32.W)))
    
    // Systolic Array 제어
    val sa_ctrl = Input(UInt(2.W))
  })

  // ===========================
  // Weight Double Buffering
  // ---------------------------
  val wgt_use  = RegInit(0.U(1.W))               // 현재 읽기 버퍼 선택
  val wgt_full = RegInit(VecInit(Seq.fill(2)(false.B)))
  val wgt_w_enable = Wire(Bool())
  when (wgt_use === 0.U) {
    wgt_w_enable := !wgt_full(1)
  } .otherwise {
    wgt_w_enable := !wgt_full(0)
  }
  io.wready := wgt_w_enable

  // 두 개의 Weight Buffer 인스턴스 생성
  val wgt_mem = Seq.fill(2)(Module(new Buffer(depth, 4, b_width)))
  when (io.wvalid && wgt_w_enable) {
    when (wgt_use === 0.U) {
      wgt_mem(1).io.wdata := io.wdata
      wgt_mem(1).io.waddr := io.waddr
      // write 시에는 해당 buffer의 enable을 true로 설정
      wgt_mem(1).io.enable := true.B
      wgt_mem(1).io.write  := true.B
      wgt_full(1) := io.wlast
    } .otherwise {
      wgt_mem(0).io.wdata := io.wdata
      wgt_mem(0).io.waddr := io.waddr
      wgt_mem(0).io.enable := true.B
      wgt_mem(0).io.write  := true.B
      wgt_full(0) := io.wlast
    }
  }
  when (io.wvalid && io.wlast && wgt_w_enable) {
    wgt_use := ~wgt_use
  }
  // Read Enable 설정 (현재 읽기 버퍼만 enable)
  val wgt_rdata = Wire(Vec(4, UInt(b_width.W)))
  when (wgt_use === 0.U) {
    wgt_mem(0).io.enable := true.B  // 읽기용 enable
    wgt_mem(1).io.enable := false.B // 비활성화
    wgt_rdata := wgt_mem(0).io.rdata
  } .otherwise {
    wgt_mem(0).io.enable := false.B
    wgt_mem(1).io.enable := true.B
    wgt_rdata := wgt_mem(1).io.rdata
  }

  // ===========================
  // Input Double Buffering
  // ---------------------------
  val inp_use  = RegInit(0.U(1.W))
  val inp_full = RegInit(VecInit(Seq.fill(2)(false.B)))
  val inp_w_enable = Wire(Bool())
  when (inp_use === 0.U) {
    inp_w_enable := !inp_full(1)
  } .otherwise {
    inp_w_enable := !inp_full(0)
  }
  io.iready := inp_w_enable

  // 두 개의 Input Buffer 인스턴스 생성
  // idata는 Vec(4, Vec(parallel, UInt(b_width.W))) 형태 → flatten하여 Vec(4*parallel, UInt(b_width.W))
  val inp_mem = Seq.fill(2)(Module(new Buffer(depth, 4 * parallel, b_width)))
  when (io.ivalid && inp_w_enable) {
    when (inp_use === 0.U) {
      inp_mem(1).io.wdata := VecInit(io.idata.flatten)
      inp_mem(1).io.waddr := io.iaddr
      inp_mem(1).io.enable := true.B
      inp_mem(1).io.write  := true.B
      inp_full(1) := io.ilast
    } .otherwise {
      inp_mem(0).io.wdata := VecInit(io.idata.flatten)
      inp_mem(0).io.waddr := io.iaddr
      inp_mem(0).io.enable := true.B
      inp_mem(0).io.write  := true.B
      inp_full(0) := io.ilast
    }
  }
  when (io.ivalid && io.ilast && inp_w_enable) {
    inp_use := ~inp_use
  }
  // Read Enable 설정: 현재 읽기 버퍼만 enable
  val inp_rdata = Wire(Vec(4 * parallel, UInt(b_width.W)))
  when (inp_use === 0.U) {
    inp_mem(0).io.enable := true.B
    inp_mem(1).io.enable := false.B
    inp_rdata := inp_mem(0).io.rdata
  } .otherwise {
    inp_mem(0).io.enable := false.B
    inp_mem(1).io.enable := true.B
    inp_rdata := inp_mem(1).io.rdata
  }

  // ===========================
  // Tag Double Buffering
  // ---------------------------
  val tag_use  = RegInit(0.U(1.W))
  val tag_full = RegInit(VecInit(Seq.fill(2)(false.B)))
  val tag_w_enable = Wire(Bool())
  when (tag_use === 0.U) {
    tag_w_enable := !tag_full(1)
  } .otherwise {
    tag_w_enable := !tag_full(0)
  }
  io.tready := tag_w_enable

  val tag_mem = Seq.fill(2)(Module(new Buffer(depth, 4, t_width + 1)))
  when (io.tvalid && tag_w_enable) {
    when (tag_use === 0.U) {
      tag_mem(1).io.wdata := io.tdata
      tag_mem(1).io.waddr := io.taddr
      tag_mem(1).io.enable := true.B
      tag_mem(1).io.write  := true.B
      tag_full(1) := io.tlast
    } .otherwise {
      tag_mem(0).io.wdata := io.tdata
      tag_mem(0).io.waddr := io.taddr
      tag_mem(0).io.enable := true.B
      tag_mem(0).io.write  := true.B
      tag_full(0) := io.tlast
    }
  }
  when (io.tvalid && io.tlast && tag_w_enable) {
    tag_use := ~tag_use
  }
  // Read Enable 설정: 현재 읽기 버퍼만 enable
  val full_rdata = Wire(Vec(4, UInt((t_width+1).W)))
  val tag_rdata = Wire(Vec(4, UInt(t_width.W)))
  val side_rdata = Wire(Vec(4, UInt(1.W)))

  when (tag_use === 0.U) {
    tag_mem(0).io.enable := true.B
    full_rdata := tag_mem(0).io.rdata
  } .otherwise {
    tag_mem(1).io.enable := true.B
  }
  
  for (i <- 0 until 4) {
    tag_rdata(i) := full_rdata(i)(t_width, 1)
    side_rdata(i) := full_rdata(i)(0)
  }

  // ===========================
  // PEStack 연결
  // ---------------------------
  val pe = Module(new PEStack(8, 4))
  // Weight 데이터 연결
  pe.io.inB := wgt_rdata
  // Input 데이터 재구성: inp_rdata를 다시 2차원 Vec(4, Vec(parallel, UInt(b_width.W)))로 변환
  pe.io.inA := VecInit((0 until 4).map { i =>
    VecInit((0 until parallel).map { j =>
      inp_rdata(i * parallel + j)
    })
  })
  // Tag 데이터 연결 (하위 t_width 비트 사용)
  pe.io.inT := tag_rdata
  // 나머지 제어 신호는 간단 예제로 처리
  pe.io.inS := side_rdata
  pe.io.inC := 0.U

  pe.io.ctrl := io.sa_ctrl

  // PEStack 결과 출력
  io.odata := pe.io.outC
}
*/

/*
class PEStack_SRAM(val b_width: Int, val parallel:Int) extends Module {
  def log2(x: Int): Int = log2Ceil(x)
  def t_width = log2(parallel)
  def s_width = 1//log2(p_width/2)

  def wgt_bf_s = 512 * 1024
  def inp_bf_s = 512 * 1024 * parallel

  val io = IO(new Bundle {
    val wdata = Input(Vec(4, UInt(b_width.W))))   // 입력 입력값
    val waddr = Input(UInt(6.W))
    val wready = Output(Bool())
    val wvalid = Input(Bool())
    val wlast  = Input(Bool())

    val idata = Input(Vec(4, Vec(parallel, UInt(b_width.W)))))   // 입력 입력값
    val iaddr = Input(UInt(6.W))
    val iready = Output(Bool())
    val ivalid = Input(Bool())
    val ilast  = Input(Bool())

    val tdata = Input(Vec(4, UInt((t_width+1).W))))   // 입력 입력값
    val taddr = Input(UInt(6.W))
    val iready = Output(Bool())
    val ivalid = Input(Bool())
    val ilast  = Input(Bool())

    val odata = Output(Vec(4, UInt(32.W))) // 다음 PE로 전달될 완성된 출력값
    val oaddr = Input(UInt(6.W))
    val ovalid = Output(Bool())
    val oready = Input(Bool())

    val sa_ctrl = Input(UInt(2.W))
  })
  
  // val wgt_use = RegInit(0.U(1.W))
  //val wgt_full = RegInit(VecInit(Seq.fill(2)(false.B)))
  //val wgt_w_enable = Wire(Bool())
  //val wlast = Mux(io.wlast, true.B, false.B)
  val wgt_mem = List.fill(2)(Module(new Buffer(32, 4, b_width)))
  
  when(wgt_use === 0.U) {
    wgt_w_enable := !wgt_full(1)
  }.otherwise {
    wgt_w_enable := !wgt_full(0)
  }
  io.wready := wgt_w_enable

  when (io.wvalid && wgt_w_enable) {
    when (wgt_use === 0.U) {
      wgt_mem(1).io.wdata := io.wdata
      wgt_mem(1).io.waddr := io.waddr
      wgt_mem(1).io.enable := true.B
      wgt_full(1) := wlast
    } .otherwise {
      wgt_mem(0).io.wdata := io.wdata
      wgt_mem(0).io.waddr := io.waddr
      wgt_mem(0).io.enable := true.B
      wgt_full(0) := wlast
    }
  }

  val tag_use = RegInit(0.U(1.W))
  val tag_mem = List.fill(2)(Module(new Buffer(32, 4, t_width + 1)))

  val inp_use = RegInit(0.U(1.W))
  val inp_mem = List.fill(2)(Module(new Buffer(32, 4*parallel, b_width)))
  
  val out_mem = List.fill(1)(Module(new Buffer(32, 1, 32)))
  
  val pe = Module(new PEStack(8, 4))

}
*/

object Mine_To_Verilog extends App {
  val rowL = 16
  val colL = 16
  val parallel = 8
  val targetDir = "/home/harry09119/chisel/src/main/scala/SA/verilog"
  val top = "Tile"
    
  if (top == "Tile") {
    val verilogFileName = s"My${top}_${rowL}_${colL}_${parallel}.v"  // 동적 파일명 생성
    println("\nGenerating Verilog of...",verilogFileName,"\n")
    (new chisel3.stage.ChiselStage).emitVerilog(
      MyTile(rowL=rowL,colL=colL,parallel = parallel),
      Array("--target-dir", targetDir, "--output-file", verilogFileName)
    )
  }
}

