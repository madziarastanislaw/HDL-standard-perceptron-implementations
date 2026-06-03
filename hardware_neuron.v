`timescale 1ns / 1ps

// Parameterized Neuron module with MAC operation and ReLU function
module hardware_neuron #(
    parameter DATA_WIDTH = 8,      // Input bit width
    parameter WEIGHT_WIDTH = 8,    // Weight bit width
    parameter ACC_WIDTH = 32       // Accumulator width
)(
    input wire clk,                     // Clock
    input wire rst_n,                   // Active-low reset
    input wire enable,                  // Enable signal
    input wire clear_acc,               // Clear accumulator
    input wire signed [DATA_WIDTH-1:0] data_in,     // Input X
    input wire signed [WEIGHT_WIDTH-1:0] weight_in, // Weight W
    input wire signed [ACC_WIDTH-1:0] bias_in,      // Bias B
    output reg signed [ACC_WIDTH-1:0] result_out,   // Result after activation
    output reg result_valid               // Result valid flag
);

    // Pipeline registers to improve clock frequency
    reg signed [DATA_WIDTH-1:0] data_reg;
    reg signed [WEIGHT_WIDTH-1:0] weight_reg;
    reg signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] mult_reg;
    reg signed [ACC_WIDTH-1:0] acc_reg;
    
    // Delay registers for control signals
    reg en_q1, en_q2, en_q3;
    reg clr_q1, clr_q2, clr_q3;

    //Input registration
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_reg   <= 0;
            weight_reg <= 0;
            en_q1      <= 0;
            clr_q1     <= 0;
        end else if (enable) begin
            data_reg   <= data_in;
            weight_reg <= weight_in;
            en_q1      <= enable;
            clr_q1     <= clear_acc;
        end else begin
            en_q1      <= 0;
        end
    end

    // Multiplication
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_reg <= 0;
            en_q2    <= 0;
            clr_q2   <= 0;
        end else begin
            // Signed multiplication
            mult_reg <= data_reg * weight_reg; 
            en_q2    <= en_q1;
            clr_q2   <= clr_q1;
        end
    end

    // Accumulation with bias
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 0;
            en_q3   <= 0;
            clr_q3  <= 0;
        end else begin
            en_q3  <= en_q2;
            clr_q3 <= clr_q2;
            
            if (clr_q2) begin
                // If clear signal is high, load bias + first multiplication
                acc_reg <= bias_in + mult_reg;
            end else if (en_q2) begin
                // Otherwise, accumulate
                acc_reg <= acc_reg + mult_reg;
            end
        end
    end

    // ReLU activation function and output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_out   <= 0;
            result_valid <= 0;
        end else begin
            // Result is valid one cycle after accumulation ends (e.g., when clear_acc is asserted for the next set)
            result_valid <= clr_q3; 
            
            // Hardware implementation of ReLU: if the sign bit (MSB) is 1 (negative), output 0. 
            // Otherwise, assign the result.
            if (acc_reg[ACC_WIDTH-1] == 1'b1) begin
                result_out <= 0;
            end else begin
                result_out <= acc_reg;
            end
        end
    end

endmodule