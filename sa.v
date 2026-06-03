`timescale 1ns / 1ps

// Systolic Array Processing Element (PE) for Matrix Multiplication in Self-Attention
module sa_systolic_pe #(
    parameter DATA_W = 8,  // Input data width
    parameter ACC_W = 32   // Accumulator width
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire clear_acc,  // Clears accumulator for a new dot product
    
    // Data flowing from the top (e.g., Query matrix Q)
    input wire signed [DATA_W-1:0] data_top,
    // Data flowing from the left (e.g., Key matrix K transpose)
    input wire signed [DATA_W-1:0] data_left,
    
    // Data forwarded to the bottom neighbor
    output reg signed [DATA_W-1:0] data_bottom,
    // Data forwarded to the right neighbor
    output reg signed [DATA_W-1:0] data_right,
    // Result of the MAC operation stored in this specific node
    output reg signed [ACC_W-1:0] acc_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_bottom <= 0;
            data_right  <= 0;
            acc_out     <= 0;
        end else if (enable) begin
            // Synchronous data propagation to neighboring PEs
            data_bottom <= data_top;
            data_right  <= data_left;
            
            // MAC operation
            if (clear_acc) begin
                // Start a new accumulation
                acc_out <= data_top * data_left;
            end else begin
                // Continue accumulating
                acc_out <= acc_out + (data_top * data_left);
            end
        end
    end
endmodule