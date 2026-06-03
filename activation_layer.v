`timescale 1ns / 1ps

// Hardware Leaky ReLU Activation Layer
module activation_layer #(
    parameter DATA_W = 32,      // Data width
    parameter ALPHA_SHIFT = 3   // Alpha approximation via bit shift
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire signed [DATA_W-1:0] data_in,
    
    output reg valid_out,
    output reg signed [DATA_W-1:0] data_out
);

    // Internal wires for logic calculation
    wire is_negative;
    wire signed [DATA_W-1:0] leaky_value;

    // The MSB defines if the signed number is negative
    assign is_negative = data_in[DATA_W-1];
    
    // Hardware-friendly multiplication by alpha (Arithmetic Shift Right)
    // Avoids using a dedicated DSP block for a simple constant multiplication
    assign leaky_value = data_in >>> ALPHA_SHIFT;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= 0;
            valid_out <= 0;
        end else begin
            // Propagate the valid signal
            valid_out <= valid_in;
            
            if (valid_in) begin
                if (is_negative) begin
                    // If negative, apply Leaky ReLU (x * alpha)
                    data_out <= leaky_value;
                end else begin
                    // If positive, pass the value unchanged (x)
                    data_out <= data_in;
                end
            end
        end
    end
endmodule