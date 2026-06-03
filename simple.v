`timescale 1ns / 1ps

module simple_combinational_perceptron (
    input wire signed [7:0] x1, x2,
    input wire signed [7:0] w1, w2,
    input wire signed [15:0] bias,
    output wire y                      
);
    wire signed [15:0] sum;

    assign sum = (x1 * w1) + (x2 * w2) + bias;

    assign y = (sum >= 0) ? 1'b1 : 1'b0;

endmodule