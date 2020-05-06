run:
	@g++ -c -Wall -Werror -fPIC -o ./source/graph_lib.o ./channel/graph_lib.cpp -fopenmp -O3 -std=c++17
	@g++ -c -Wall -Werror -fPIC -o ./source/graph.o ./channel/graph.cpp -fopenmp -O3 -std=c++17
	@g++ -c -Wall -Werror -fPIC -o ./source/heap.o ./channel/heap.cpp -fopenmp -O3 -std=c++17
	@g++ -Wall -Werror -shared ./source/graph_lib.o ./source/graph.o ./source/heap.o -o ./source/lib_graph.so -fopenmp
	@python3 main.py
	@rm ./source/graph_lib.o ./source/heap.o ./source/graph.o ./source/lib_graph.so
