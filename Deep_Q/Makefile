run:
	@g++ -c -Wall -Werror -fPIC -o ./source/graph_lib.o ./channel/graph_lib.cpp -fopenmp -O3 -std=c++17
	@g++ -shared ./source/graph_lib.o -o ./source/lib_graph.so -fopenmp
	@python main.py
#	@del .\source\graph_lib.o .\source\lib_graph.so
	@rm ./source/graph_lib.o ./source/lib_graph.so