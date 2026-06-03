`timescale 1ns / 1ps

// Pipelined 3x3 Convolution MAC Engine
module cnn_3x3_engine #(
    parameter DATA_W = 8,   // Input data width (e.g., INT8)
    parameter WEIGHT_W = 8, // Weight width (e.g., INT8)
    parameter MULT_W = 16,  // Multiplier output width
    parameter ACC_W = 32    // Accumulator width
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    
    // 9 pixels from the sliding window (e.g., from a Line Buffer)
    input wire signed [DATA_W-1:0] p1, p2, p3, p4, p5, p6, p7, p8, p9,
    // 9 filter weights
    input wire signed [WEIGHT_W-1:0] w1, w2, w3, w4, w5, w6, w7, w8, w9,
    input wire signed [ACC_W-1:0] bias,
    
    output reg valid_out,
    output reg signed [ACC_W-1:0] conv_out // Raw convolution result (before activation)
);

    // Valid signal pipeline registers to match the data path latency
    reg val_q1, val_q2, val_q3;

    // Parallel Multiplication (Maps to 9 DSP blocks)
    reg signed [MULT_W-1:0] m1, m2, m3, m4, m5, m6, m7, m8, m9;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1 <= 0; m2 <= 0; m3 <= 0; m4 <= 0; m5 <= 0;
            m6 <= 0; m7 <= 0; m8 <= 0; m9 <= 0;
            val_q1 <= 0;
        end else if (valid_in) begin
            // Perform 9 multiplications simultaneously
            m1 <= p1 * w1; m2 <= p2 * w2; m3 <= p3 * w3;
            m4 <= p4 * w4; m5 <= p5 * w5; m6 <= p6 * w6;
            m7 <= p7 * w7; m8 <= p8 * w8; m9 <= p9 * w9;
            val_q1 <= 1'b1;
        end else begin
            val_q1 <= 0;
        end
    end

    // Adder Tree - Level 1 (Pairwise addition to reduce path delay)
    reg signed [MULT_W:0] sum1_1, sum1_2, sum1_3, sum1_4, sum1_5;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum1_1 <= 0; sum1_2 <= 0; sum1_3 <= 0; sum1_4 <= 0; sum1_5 <= 0;
            val_q2 <= 0;
        end else begin
            sum1_1 <= m1 + m2;
            sum1_2 <= m3 + m4;
            sum1_3 <= m5 + m6;
            sum1_4 <= m7 + m8;
            sum1_5 <= m9; // Passes through to the next stage
            val_q2 <= val_q1;
        end
    end

    // Adder Tree - Level 2
    reg signed [MULT_W+1:0] sum2_1, sum2_2, sum2_3;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum2_1 <= 0; sum2_2 <= 0; sum2_3 <= 0;
            val_q3 <= 0;
        end else begin
            sum2_1 <= sum1_1 + sum1_2;
            sum2_2 <= sum1_3 + sum1_4;
            sum2_3 <= sum1_5;
            val_q3 <= val_q2;
        end
    end

    // Final Accumulation + Bias
    wire signed [ACC_W-1:0] final_sum;
    assign final_sum = sum2_1 + sum2_2 + sum2_3 + bias;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_out <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= val_q3;
            if (val_q3) begin
                conv_out <= final_sum;
            end
        end
    end
endmodule