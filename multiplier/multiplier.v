module multiplier #(
    parameter A_WIDTH = 8,
    parameter B_WIDTH = 8
)(
    input  [A_WIDTH-1:0] a,
    input  [B_WIDTH-1:0] b,
    output [A_WIDTH+B_WIDTH-1:0] product
);
    assign product = a * b;
endmodule
