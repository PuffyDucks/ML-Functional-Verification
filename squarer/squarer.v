module squarer (
    input [7:0] a,
    output [15:0] result
);

    assign result = a * a;

endmodule
