module comparator #(parameter WIDTH = 5)(
    input [WIDTH-1:0] a, b,
    output reg gt, lt, eq
);
    always @(*) begin
        gt = (a > b);
        lt = (a < b);
        eq = (a == b);
    end
endmodule