# Makefile for FFI examples

CC = gcc
CFLAGS = -Wall -Wextra -I../../include -O2
LDFLAGS = -L../.. -lmojo_audio -lm

RUSTC = rustc
RUST_FLAGS = -L ../.. -l mojo_audio

all: demo_c demo_rust

demo_c: demo_c.c
	$(CC) $(CFLAGS) demo_c.c $(LDFLAGS) -o demo_c

demo_rust: demo_rust.rs
	$(RUSTC) $(RUST_FLAGS) demo_rust.rs -o demo_rust

clean:
	rm -f demo_c demo_rust

run_c: demo_c
	LD_LIBRARY_PATH=../.. ./demo_c

run_rust: demo_rust
	LD_LIBRARY_PATH=../.. ./demo_rust

.PHONY: all clean run_c run_rust
