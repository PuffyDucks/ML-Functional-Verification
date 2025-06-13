module polynomial (
    input  wire [9:0] a,
    output wire [15:0] result
);

    // result = ((a-600)(a-940)(a) / 2^15) + 700
    assign result = ((((a * ($signed({6'd0, a}) - 600) * ($signed({6'd0, a}) - 940)) >>> 15)) + 700);

endmodule
