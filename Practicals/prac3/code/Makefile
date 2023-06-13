# Compiler configuration
CC=gcc		# C Compiler: GCC
INCDIRS=-I.	# Include Directories: current directory
OPT=-O0		# Optimisation level: 0
CFLAGS=-Wall -Wextra -g $(INCDIRS) $(OPT)	# Compiler flags: All warnings, Extra warnings, Include debug info, Set include directories, Set optimisation level

# Files
CFILES=barley.c parse.c shapes.c		# Source files
OBJECTS=barley.o parse.o shapes.o	# Object files (compiled)
BINARY=barley	# Executable file name (output of linker)

# Targets
all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

clean:
	rm -rf $(BINARY) $(OBJECTS)