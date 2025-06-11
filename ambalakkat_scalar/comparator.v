module comparator #(parameter WIDTH = 4)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output reg [WIDTH*2-1:0] match
);

    always @(*) begin
        match = (a==b) ? a*b : 0;
    end

endmodule
