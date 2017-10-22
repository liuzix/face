CC=nvcc
CFLAGS= -x cu -std=c++11 -dc -g -G -O0 --expt-extended-lambda
LDFLAGS= -ljpeg -L/opt/X11/lib -lX11

SOURCES = main.cpp jpeg.cpp feature.cpp adaboost.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = face

# Default make option
all: $(SOURCES) $(EXECUTABLE)

# Note the position of LDFLAGS    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) -arch=sm_61 $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -arch=sm_61 -c $< -o $@

clean:
	rm *.o face