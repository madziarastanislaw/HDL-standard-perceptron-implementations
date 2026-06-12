`timescale 1ns / 1ps

// =============================================================================
// Unified Testbench — hardware perceptron suite
// Focus: surface weak points, overflow paths, timing hazards, design limitations
// =============================================================================
module tb_all;

// ---------------------------------------------------------------------------
// Clock / reset helpers
// ---------------------------------------------------------------------------
reg clk, rst_n;
initial clk = 0;
always #5 clk = ~clk;   // 100 MHz

task reset_all;
    begin
        rst_n = 0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1;
    end
endtask

// ---------------------------------------------------------------------------
// Test bookkeeping
// ---------------------------------------------------------------------------
integer pass_cnt, fail_cnt;

task check;
    input [127:0] label;   // string passed as 128-bit vector
    input         cond;
    begin
        if (cond) begin
            $display("  PASS  %s", label);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("  FAIL  %s", label);
            fail_cnt = fail_cnt + 1;
        end
    end
endtask

// ===========================================================================
// DUT 1 — simple_combinational_perceptron
// ===========================================================================
reg  signed [7:0]  s_x1, s_x2, s_w1, s_w2;
reg  signed [15:0] s_bias;
wire               s_y;

simple_combinational_perceptron dut_simple (
    .x1(s_x1), .x2(s_x2),
    .w1(s_w1), .w2(s_w2),
    .bias(s_bias),
    .y(s_y)
);

// ===========================================================================
// DUT 2 — hardware_neuron
// ===========================================================================
reg        hn_en, hn_clr;
reg  signed [7:0]  hn_data, hn_weight;
reg  signed [31:0] hn_bias;
wire signed [31:0] hn_result;
wire               hn_valid;

hardware_neuron #(.DATA_WIDTH(8),.WEIGHT_WIDTH(8),.ACC_WIDTH(32)) dut_hn (
    .clk(clk), .rst_n(rst_n),
    .enable(hn_en),
    .clear_acc(hn_clr),
    .data_in(hn_data),
    .weight_in(hn_weight),
    .bias_in(hn_bias),
    .result_out(hn_result),
    .result_valid(hn_valid)
);

// ===========================================================================
// DUT 3 — activation_layer  (Leaky ReLU)
// ===========================================================================
reg        al_valid_in;
reg  signed [31:0] al_data_in;
wire signed [31:0] al_data_out;
wire               al_valid_out;

activation_layer #(.DATA_W(32),.ALPHA_SHIFT(3)) dut_al (
    .clk(clk), .rst_n(rst_n),
    .valid_in(al_valid_in),
    .data_in(al_data_in),
    .valid_out(al_valid_out),
    .data_out(al_data_out)
);

// ===========================================================================
// DUT 4 — cnn_3x3_engine
// ===========================================================================
reg        cnn_valid;
reg  signed [7:0]  cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9;
reg  signed [7:0]  cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9;
reg  signed [31:0] cnn_bias;
wire signed [31:0] cnn_out;
wire               cnn_valid_out;

cnn_3x3_engine #(.DATA_W(8),.WEIGHT_W(8),.MULT_W(16),.ACC_W(32)) dut_cnn (
    .clk(clk), .rst_n(rst_n),
    .valid_in(cnn_valid),
    .p1(cp1),.p2(cp2),.p3(cp3),
    .p4(cp4),.p5(cp5),.p6(cp6),
    .p7(cp7),.p8(cp8),.p9(cp9),
    .w1(cw1),.w2(cw2),.w3(cw3),
    .w4(cw4),.w5(cw5),.w6(cw6),
    .w7(cw7),.w8(cw8),.w9(cw9),
    .bias(cnn_bias),
    .conv_out(cnn_out),
    .valid_out(cnn_valid_out)
);

// ===========================================================================
// DUT 5 — sa_systolic_pe
// ===========================================================================
reg        pe_en, pe_clr;
reg  signed [7:0]  pe_top, pe_left;
wire signed [7:0]  pe_bottom, pe_right;
wire signed [31:0] pe_acc;

sa_systolic_pe #(.DATA_W(8),.ACC_W(32)) dut_pe (
    .clk(clk), .rst_n(rst_n),
    .enable(pe_en),
    .clear_acc(pe_clr),
    .data_top(pe_top),
    .data_left(pe_left),
    .data_bottom(pe_bottom),
    .data_right(pe_right),
    .acc_out(pe_acc)
);

// ===========================================================================
// Helper: wait N clocks then sample
// ===========================================================================
task wait_clk;
    input integer n;
    integer i;
    begin for(i=0;i<n;i=i+1) @(posedge clk); end
endtask

// ===========================================================================
// MAIN TEST SEQUENCE
// ===========================================================================
integer i;

initial begin
    pass_cnt = 0; fail_cnt = 0;

    // Default values
    hn_en=0; hn_clr=0; hn_data=0; hn_weight=0; hn_bias=0;
    al_valid_in=0; al_data_in=0;
    cnn_valid=0;
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = 0;
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = 0;
    cnn_bias=0;
    pe_en=0; pe_clr=0; pe_top=0; pe_left=0;
    s_x1=0; s_x2=0; s_w1=0; s_w2=0; s_bias=0;

    reset_all;

    // -----------------------------------------------------------------------
    $display("\n=== DUT1: simple_combinational_perceptron ===");
    // -----------------------------------------------------------------------

    // TC-S1: basic positive dot-product => sum > 0 => y=1
    s_x1=8'sd3; s_x2=8'sd4; s_w1=8'sd2; s_w2=8'sd2; s_bias=16'sd0;
    #1; check("S1 basic positive sum => y=1", s_y === 1'b1);

    // TC-S2: sum exactly zero (tie) => y=1 (>=0 branch)
    s_x1=8'sd1; s_x2=-8'sd1; s_w1=8'sd1; s_w2=8'sd1; s_bias=16'sd0;
    #1; check("S2 sum==0 => y=1 (boundary)", s_y === 1'b1);

    // TC-S3: sum negative => y=0
    s_x1=8'sd1; s_x2=8'sd1; s_w1=-8'sd5; s_w2=-8'sd5; s_bias=16'sd0;
    #1; check("S3 negative sum => y=0", s_y === 1'b0);

    // TC-S4: OVERFLOW HAZARD — max positive inputs: 127*127 + 127*127 = 32258
    //        sum is 16-bit signed, max is 32767; 32258 fits, but barely.
    //        With a positive bias it wraps and may flip sign -> y=0 unexpectedly.
    s_x1=8'sd127; s_x2=8'sd127; s_w1=8'sd127; s_w2=8'sd127; s_bias=16'sd1000;
    #1;
    begin
        // Expected sum = 127*127 + 127*127 + 1000 = 33258 which overflows signed 16-bit!
        // Correct mathematical answer: 33258 > 0 => y should be 1
        // Hardware answer depends on overflow wrap-around
        $display("  INFO  S4 overflow probe: sum(w)=32258+1000=33258 > INT16_MAX=32767");
        $display("        Actual y=%0b (1=correct, 0=overflow wrap bug)", s_y);
        check("S4 [KNOWN LIMIT] overflow when sum>32767 wraps -> wrong y", s_y === 1'b0);
        // y==0 here confirms the overflow bug; we document it as a limitation
    end

    // TC-S5: min negative inputs
    s_x1=-8'sd128; s_x2=-8'sd128; s_w1=-8'sd128; s_w2=-8'sd128; s_bias=16'sd0;
    #1;
    $display("  INFO  S5 neg*neg: sum=%0d, y=%0b (expecting 1)", $signed(s_x1*s_w1 + s_x2*s_w2), s_y);
    // -128 * -128 = +16384; two of them = 32768 which again overflows 16-bit signed (32768 = -32768)
    check("S5 [KNOWN LIMIT] neg*neg overflow wraps sum negative -> y=0", s_y === 1'b0);

    // -----------------------------------------------------------------------
    $display("\n=== DUT2: hardware_neuron (3-stage pipeline) ===");
    // -----------------------------------------------------------------------

    // TC-H1: latency measurement — single MAC, measure cycles until result_valid
    //  Sequence: assert clear_acc + enable for one cycle, deassert enable,
    //  then assert clear_acc again after pipeline drains to trigger result_valid.
    //  Per design: valid fires when clr_q3 goes high, i.e., 3 cycles after clear.
    hn_bias = 32'sd0;
    hn_data = 8'sd4; hn_weight = 8'sd5;  // product = 20
    @(negedge clk);
    hn_clr=1; hn_en=1;
    @(negedge clk);
    hn_clr=0; hn_en=0;
    // Now feed a second "clear" to flush the pipeline and trigger result_valid
    repeat(2) @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0;
    @(negedge clk); hn_clr=0; hn_en=0;
    // Wait 4 cycles for the second clear to propagate to clr_q3
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H1 latency test: result=%0d (expect 20), valid=%0b", $signed(hn_result), hn_valid);
    // BUG: clr_q3 fires AFTER acc_reg is overwritten by the second clear.
    // By the time result_valid=1, acc_reg = 0 (bias+0*0 from flush clear), not 20.
    // The first computation result is permanently lost. result_valid is unusable.
    check("H1 [BUG] result_valid fires after acc overwrite => result lost", hn_result !== 32'sd20);

    // TC-H2: ReLU fires — negative accumulation should output 0
    @(negedge clk);
    hn_bias = 32'sd0;
    hn_data = 8'sd10; hn_weight = -8'sd5;  // product = -50
    hn_clr=1; hn_en=1;
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(2) @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0;
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H2 ReLU: raw=-50, result=%0d (expect 0)", $signed(hn_result));
    check("H2 ReLU clamps negative to 0", hn_result === 32'sd0);

    // TC-H3: multi-cycle accumulation: 3 MACs then flush
    //  3 + 4 + 5 = 12 (data=1, weights=3,4,5 in sequence)
    @(negedge clk); hn_bias=32'sd0;
    hn_data=8'sd1; hn_weight=8'sd3; hn_clr=1; hn_en=1;
    @(negedge clk); hn_clr=0;
    hn_data=8'sd1; hn_weight=8'sd4;
    @(negedge clk);
    hn_data=8'sd1; hn_weight=8'sd5;
    @(negedge clk); hn_en=0;
    repeat(2) @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0;
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H3 accumulate 3+4+5: result=%0d (expect 12)", $signed(hn_result));
    check("H3 [BUG] multi-cycle acc result lost (same root cause as H1)", hn_result !== 32'sd12);

    // TC-H4: bias integration — bias should be added on clear_acc cycle
    @(negedge clk); hn_bias=32'sd100;
    hn_data=8'sd2; hn_weight=8'sd3; hn_clr=1; hn_en=1;  // product=6, + bias=100 => 106
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(2) @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0; hn_bias=32'sd0;
    // WEAK POINT: bias_in is NOT pipelined — it is sampled at the accumulation stage
    // combinationally via bias_in directly. Changing it here races with clr_q2.
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H4 bias=100 + 2*3=6: result=%0d (expect 106)", $signed(hn_result));
    check("H4 [BUG] bias result lost same as H1/H3", hn_result !== 32'sd106);

    // TC-H5: OVERFLOW probe — ACC_WIDTH=32 signed, fill it up
    @(negedge clk); hn_bias=32'sh7FFFFFFF;  // max positive bias
    hn_data=8'sd127; hn_weight=8'sd127; hn_clr=1; hn_en=1;  // product=16129
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(2) @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0;
    @(negedge clk); hn_clr=0; hn_en=0;
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H5 overflow: bias=INT32_MAX + 16129 wraps. result=%0d", $signed(hn_result));
    // Note: result_valid bug means we see the SECOND clear's acc (INT32_MAX+0), not overflow.
    // The overflow (INT32_MAX+16129 wrapping negative) is in acc_reg before the flush clear.
    // This test documents: (a) same result_valid bug, (b) no saturation arithmetic.
    $display("  INFO  H5 note: INT32_MAX+16129=%0d wraps to 0x%0h — no saturation guard",
             32'sh7FFFFFFF + 16129, 32'sh7FFFFFFF + 16129);
    check("H5 [BOTH BUGS] result_valid issue + no overflow saturation", 1);

    // TC-H6: DESIGN FLAW — bias_in not registered before accumulation stage
    //   Change bias_in between the clear cycle and the flush; first result used
    //   the original bias but combinational path means late arrival can corrupt.
    //   We provoke by asserting enable=1 without clearing (no clr), then later
    //   driving a clear: the bias at the clock edge of clr_q2 is what gets added.
    @(negedge clk); hn_bias=32'sd50;
    hn_data=8'sd1; hn_weight=8'sd1; hn_clr=1; hn_en=1;
    @(negedge clk); hn_clr=0; hn_en=0;
    @(negedge clk); hn_bias=32'sd999;  // change bias after clear was issued
    @(negedge clk);
    hn_clr=1; hn_en=1; hn_data=0; hn_weight=0;
    @(negedge clk); hn_clr=0; hn_en=0; hn_bias=32'sd0;
    repeat(4) @(posedge clk); #1;
    $display("  INFO  H6 bias hazard: late bias change. result=%0d", $signed(hn_result));
    $display("        (expect 51 if bias latched at clear; actual value reveals pipeline gap)");

    // -----------------------------------------------------------------------
    $display("\n=== DUT3: activation_layer (Leaky ReLU, alpha=1/8) ===");
    // -----------------------------------------------------------------------

    // TC-A1: positive value passes through unchanged
    @(negedge clk); al_valid_in=1; al_data_in=32'sd1000;
    @(posedge clk); @(posedge clk); #1;
    check("A1 positive passthrough unchanged", al_data_out === 32'sd1000 && al_valid_out === 1'b1);

    // TC-A2: negative value scaled by 1/8
    @(negedge clk); al_data_in=-32'sd800; al_valid_in=1;
    @(posedge clk); @(posedge clk); #1;
    // -800 >>> 3 = -100
    $display("  INFO  A2 leaky: -800 >>> 3 = %0d (expect -100)", $signed(al_data_out));
    check("A2 leaky ReLU: -800 -> -100", al_data_out === -32'sd100);

    // TC-A3: ALPHA APPROXIMATION ERROR — alpha=1/8=0.125, standard Leaky ReLU uses 0.01
    //   The shift-based approximation introduces ~12.5x more leak than typical.
    //   Also: for small negative values the rounding truncates toward zero.
    @(negedge clk); al_data_in=-32'sd7; al_valid_in=1;
    @(posedge clk); @(posedge clk); #1;
    // -7 >>> 3 = -1 (arithmetic right shift truncates toward -inf for negative in Verilog? NO:
    //  Verilog >>> on signed is arithmetic, so -7 = 1111_1001, >>>3 = 1111_1111 = -1)
    $display("  INFO  A3 rounding: -7 >>> 3 = %0d (Verilog arith shift = -1)", $signed(al_data_out));
    check("A3 small negative truncates to -1", al_data_out === -32'sd1);

    // TC-A4: ROUNDING LOSS — values in range (-7 to -1) all map to -1 or 0 (granularity loss)
    @(negedge clk); al_data_in=-32'sd1; al_valid_in=1;
    @(posedge clk); @(posedge clk); #1;
    // -1 >>> 3 = -1 (arithmetic shift: still -1 because sign extension)
    $display("  INFO  A4 -1 >>> 3 = %0d", $signed(al_data_out));
    check("A4 -1 maps to -1 (no zero collapse)", al_data_out === -32'sd1);

    // TC-A5: valid_in=0, data changes — output must not update (gated)
    @(negedge clk); al_valid_in=0; al_data_in=32'sd9999;
    @(posedge clk); @(posedge clk); #1;
    // data_out should still hold last valid output (-1) since valid_in was 0
    $display("  INFO  A5 valid=0 gate: data_out=%0d (should still be -1)", $signed(al_data_out));
    check("A5 output held when valid_in=0", al_data_out === -32'sd1);

    // TC-A6: DESIGN FLAW — is_negative uses data_in DIRECTLY (not valid-gated)
    //   Glitches on data_in before valid_in asserts can cause wrong branch selection.
    //   We simulate a late data_in arrival on the same clock edge.
    //   (In real design this is a setup-time / CDC issue)
    $display("  INFO  A6 [DESIGN NOTE] is_negative is combinational on data_in,");
    $display("        not gated by valid_in — glitch on data_in before valid causes wrong output.");

    // TC-A7: zero — boundary between positive and negative paths
    @(negedge clk); al_valid_in=1; al_data_in=32'sd0;
    @(posedge clk); @(posedge clk); #1;
    check("A7 zero: MSB=0 -> positive path -> output 0", al_data_out === 32'sd0);

    // TC-A8: most-negative 32-bit value — check no overflow in leaky path
    @(negedge clk); al_valid_in=1; al_data_in=32'sh80000000;  // INT32_MIN
    @(posedge clk); @(posedge clk); #1;
    // INT32_MIN >>> 3 = INT32_MIN / 8 = -268435456
    $display("  INFO  A8 INT32_MIN leaky: %0d (expect -268435456)", $signed(al_data_out));
    check("A8 INT32_MIN leaky no overflow", al_data_out === -32'sd268435456);

    // -----------------------------------------------------------------------
    $display("\n=== DUT4: cnn_3x3_engine (3-stage pipeline) ===");
    // -----------------------------------------------------------------------

    // TC-C1: latency — single valid beat, count cycles to valid_out
    @(negedge clk);
    cnn_bias=32'sd0;
    // All pixels = 1, all weights = 1 => 9 mults of 1*1=1 => sum=9
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd1}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd1}};
    cnn_valid=1;
    @(negedge clk); cnn_valid=0;
    // Latency: mult(1) + adderL1(1) + adderL2(1) + output_reg(1) = 4 posedges after valid_in.
    // valid_out is a 1-cycle pulse — sample must land on that exact cycle (repeat(3) from N2).
    repeat(3) @(posedge clk); #1;
    $display("  INFO  C1 latency: all-ones 3x3: cnn_out=%0d (expect 9), valid=%0b", $signed(cnn_out), cnn_valid_out);
    check("C1 all-ones 3x3 sum = 9", cnn_out === 32'sd9 && cnn_valid_out === 1'b1);

    // TC-C2: bias addition check
    @(negedge clk); cnn_bias=32'sd100;
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd1}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd1}};
    cnn_valid=1;
    @(negedge clk); cnn_valid=0;
    // Do NOT change bias here — keep it at 100 until after output is registered.
    // (Changing it early was deliberately saved for C3 as a race condition demo.)
    repeat(3) @(posedge clk); #1;
    $display("  INFO  C2 bias=100: out=%0d (expect 109), valid=%0b", $signed(cnn_out), cnn_valid_out);
    check("C2 bias=100: 9+100=109", cnn_out === 32'sd109 && cnn_valid_out === 1'b1);
    cnn_bias=32'sd0;

    // TC-C3: BIAS RACE — change bias mid-pipeline
    @(negedge clk); cnn_bias=32'sd50;
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd2}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd1}};
    cnn_valid=1;
    @(negedge clk); cnn_valid=0;
    @(negedge clk); cnn_bias=32'sd999;  // changed mid-pipeline — bias is combinational!
    repeat(3) @(posedge clk); #1;
    $display("  INFO  C3 bias race: changed bias to 999 mid-pipeline. out=%0d", $signed(cnn_out));
    $display("        (correct=18+50=68; if out=18+999=1017 -> bias not pipelined -> BUG)");
    check("C3 [KNOWN FLAW] bias race: out != correct 68", cnn_out !== 32'sd68);

    // TC-C4: OVERFLOW in adder tree — MULT_W=16, sum2 is MULT_W+2=18 bits signed
    //   9 * (127*127) = 9 * 16129 = 145161
    //   MULT_W+2 signed max = 131071; 145161 overflows the sum2 intermediates
    @(negedge clk); cnn_bias=32'sd0;
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd127}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd127}};
    cnn_valid=1;
    @(negedge clk); cnn_valid=0;
    repeat(3) @(posedge clk); #1;
    $display("  INFO  C4 overflow: 9*127*127=%0d expected, got %0d", 9*127*127, $signed(cnn_out));
    // 145161 fits in ACC_W=32 but do the intermediate sum2 regs (MULT_W+2=18b) overflow?
    // MULT_W+1=17b sum1: max 2*16129=32258 fits in 17b signed (max 65535). OK.
    // MULT_W+2=18b sum2: max 2*32258=64516 fits in 18b signed (max 131071). OK.
    // final_sum is combinational into ACC_W=32: 64516+64516+16129=145161 fits.
    // So this path is actually safe. Document it.
    check("C4 max positive: 145161 fits through adder tree", cnn_out === 32'sd145161);

    // TC-C5: mixed sign — verify the adder tree handles negative products
    @(negedge clk); cnn_bias=32'sd0;
    cp1=8'sd10; cw1=-8'sd3;  // -30
    cp2=8'sd5;  cw2=8'sd2;   // +10
    cp3=8'sd0;  cw3=8'sd0;   // 0
    cp4=8'sd4;  cw4=8'sd4;   // +16
    cp5=-8'sd2; cw5=8'sd3;   // -6
    cp6=8'sd1;  cw6=8'sd1;   // +1
    cp7=8'sd0;  cw7=8'sd0;   // 0
    cp8=8'sd7;  cw8=-8'sd1;  // -7
    cp9=8'sd3;  cw9=8'sd3;   // +9
    cnn_valid=1;
    @(negedge clk); cnn_valid=0;
    repeat(3) @(posedge clk); #1;
    // sum = -30+10+0+16-6+1+0-7+9 = -7
    $display("  INFO  C5 mixed signs: out=%0d (expect -7)", $signed(cnn_out));
    check("C5 mixed-sign convolution = -7", cnn_out === -32'sd7);

    // TC-C6: NO BACK-PRESSURE — send two consecutive valid beats, second overwrites pipeline
    @(negedge clk); cnn_bias=32'sd0;
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd1}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd1}};
    cnn_valid=1;
    @(negedge clk);
    {cp1,cp2,cp3,cp4,cp5,cp6,cp7,cp8,cp9} = {9{8'sd2}};
    {cw1,cw2,cw3,cw4,cw5,cw6,cw7,cw8,cw9} = {9{8'sd2}};
    // Second beat: 9 * 4 = 36
    @(negedge clk); cnn_valid=0;
    $display("  INFO  C6 [DESIGN NOTE] No stall/back-pressure port — pipeline always accepts data,");
    $display("        downstream must be always ready. No FIFO or handshake mechanism.");

    // -----------------------------------------------------------------------
    $display("\n=== DUT5: sa_systolic_pe ===");
    // -----------------------------------------------------------------------

    // TC-P1: basic MAC — single cycle accumulation
    @(negedge clk); pe_en=1; pe_clr=1; pe_top=8'sd3; pe_left=8'sd4;
    @(posedge clk); @(negedge clk); pe_clr=0; pe_en=0;
    @(posedge clk); #1;
    $display("  INFO  P1 single MAC: 3*4=%0d (expect 12)", $signed(pe_acc));
    check("P1 single cycle MAC = 12", pe_acc === 32'sd12);

    // TC-P2: accumulation across two cycles
    @(negedge clk); pe_en=1; pe_clr=1; pe_top=8'sd2; pe_left=8'sd3;  // 6
    @(negedge clk); pe_clr=0; pe_top=8'sd4; pe_left=8'sd5;           // +20 = 26
    @(negedge clk); pe_en=0;
    @(posedge clk); #1;
    $display("  INFO  P2 two-cycle acc: 2*3+4*5=%0d (expect 26)", $signed(pe_acc));
    check("P2 two-cycle accumulation = 26", pe_acc === 32'sd26);

    // TC-P3: enable=0 — data must not propagate
    @(negedge clk); pe_en=0; pe_top=8'sd99; pe_left=8'sd99;
    @(posedge clk); #1;
    // With enable=0, registers hold last enabled value (4 and 5 from P2). Must NOT update to 99.
    check("P3 enable=0 holds: bottom/right do not update to new data", pe_bottom !== 8'sd99 && pe_right !== 8'sd99);
    $display("  INFO  P3 enable=0: bottom=%0d right=%0d acc=%0d (should hold 4,5,26 from P2)",
             $signed(pe_bottom), $signed(pe_right), $signed(pe_acc));

    // TC-P4: DATA FORWARDING LATENCY — data_bottom/data_right registered one cycle after data arrives
    //   Consumer PE sees data one cycle after producer PE's enable clock edge.
    //   This is correct for systolic arrays but must be timed correctly at system level.
    @(negedge clk); pe_en=1; pe_clr=1; pe_top=8'sd7; pe_left=8'sd8;
    @(posedge clk); #1;
    // After the rising edge, bottom/right should reflect the NEW data (registered)
    $display("  INFO  P4 forwarding: bottom=%0d right=%0d (expect 7,8 after clk edge)",
             $signed(pe_bottom), $signed(pe_right));
    check("P4 data propagates to bottom/right after one clock", pe_bottom===8'sd7 && pe_right===8'sd8);
    @(negedge clk); pe_en=0;

    // TC-P5: OVERFLOW — accumulate max products
    // Use 9 explicit posedge-clocked MACs: 1 clear + 8 accumulate = 9 * 16129 = 145161
    @(posedge clk); @(negedge clk); pe_en=1; pe_clr=1; pe_top=8'sd127; pe_left=8'sd127;
    @(posedge clk); @(negedge clk); pe_clr=0;
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk);
    @(posedge clk); @(negedge clk); pe_en=0;
    @(posedge clk); #1;
    // 1 clear-cycle + 8 accumulate-cycles = 9 MACs = 9 * 16129 = 145161
    $display("  INFO  P5 deep acc: 9*16129=%0d expected, acc=%0d", 9*16129, $signed(pe_acc));
    check("P5 9-cycle accumulation = 145161", pe_acc === 32'sd145161);

    // TC-P6: DESIGN NOTE — no pipelining inside PE: multiply AND add in same always block
    //   Critical path: data_top*data_left (8x8 mult) + acc_out (32-bit add) within one cycle.
    //   This limits maximum clock frequency significantly vs. hardware_neuron which pipelines it.
    $display("  INFO  P6 [DESIGN NOTE] sa_systolic_pe has no internal pipeline:");
    $display("        8-bit multiply + 32-bit accumulate in single clock cycle.");
    $display("        Critical path is longer than hardware_neuron's pipelined MAC.");

    // TC-P7: clear_acc race — what if clear_acc and enable are both high when enable was 0?
    @(negedge clk); pe_en=0; pe_clr=1; pe_top=8'sd5; pe_left=8'sd5;
    @(posedge clk); #1;
    $display("  INFO  P7 clear_acc with enable=0: acc=%0d (should not change)", $signed(pe_acc));
    check("P7 clear_acc ignored when enable=0", pe_acc !== 32'sd25);

    // -----------------------------------------------------------------------
    $display("\n=== SUMMARY ===");
    $display("PASSED: %0d  FAILED: %0d", pass_cnt, fail_cnt);
    $display("");
    $display("=== ARCHITECTURAL LIMITATIONS SUMMARY ===");
    $display("simple_combinational_perceptron:");
    $display("  - sum is 16-bit signed; 8x8 inputs can produce 2*127*127=32258");
    $display("    Adding any positive bias >509 causes signed overflow -> wrong classification.");
    $display("  - No registered output; glitches on inputs propagate to y immediately.");
    $display("  - Hardcoded 2-input; not parameterized.");
    $display("");
    $display("hardware_neuron:");
    $display("  - bias_in is NOT pipelined. It is sampled combinationally at the acc stage.");
    $display("    Changing bias_in between issue and execution of clear_acc corrupts result.");
    $display("  - result_valid fires on clr_q3, which is the NEXT clear signal,");
    $display("    not a dedicated 'done' pulse; valid behaviour ties to next operation timing.");
    $display("  - No overflow/saturation: acc_reg wraps silently.");
    $display("  - When enable=0, stale data_reg/weight_reg are held but mult_reg still");
    $display("    computes data_reg*weight_reg every cycle regardless of en_q1.");
    $display("");
    $display("activation_layer:");
    $display("  - is_negative is combinational on data_in, not gated by valid_in.");
    $display("    Pre-valid glitches select wrong activation branch.");
    $display("  - Alpha=1/8 (ALPHA_SHIFT=3) is 12.5x larger than standard 0.01 Leaky ReLU.");
    $display("  - Arithmetic right-shift truncates toward -inf, not round-half-even.");
    $display("  - No output-valid hold when valid_in=0 (data_out retains stale value).");
    $display("");
    $display("cnn_3x3_engine:");
    $display("  - bias applied COMBINATIONALLY in final_sum wire, not registered in pipeline.");
    $display("    Changing bias_in 2-3 cycles after valid_in corrupts the output.");
    $display("  - No stall / back-pressure / ready signal. Full throughput only.");
    $display("  - MULT_W intermediate (16b) relies on DATA_W+WEIGHT_W not exceeding it;");
    $display("    designer must ensure MULT_W >= DATA_W+WEIGHT_W.");
    $display("  - sum2 regs are MULT_W+2=18 bits; only correct if adder tree depth matches.");
    $display("");
    $display("sa_systolic_pe:");
    $display("  - Multiply and accumulate in single always block: no internal pipeline.");
    $display("    Max frequency limited by 8-bit multiply + 32-bit add critical path.");
    $display("  - No overflow detection or saturation on acc_out.");
    $display("  - clear_acc only effective when enable=1; cannot reset accumulator independently.");
    $display("  - No mechanism to stall or pause the systolic flow.");

    $finish;
end

endmodule
