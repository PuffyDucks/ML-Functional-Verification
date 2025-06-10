TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(PWD)/comparator.v
TOPLEVEL = comparator
MODULE = test_comparator
TESTCASE = run_batch

SIM = verilator
EXTRA_ARGS ?=

include $(shell cocotb-config --makefiles)/Makefile.sim
