NAME		:= Vulkan

SRCS		:= srcs/App.cpp

DIR_SRCS	:= srcs
DIR_OBJS	:= .objs
DEP			:= dependencies

OBJS		:= $(addprefix $(DIR_OBJS)/, $(notdir $(SRCS:.cpp=.o)))

CC			:= c++
CFLAGS		:= -Wall -Wextra -std=c++17 -g
#CFLAGS		:= -std=c++17 -g
IFLAGS		:= -I include -I $(DEP)/glm -I $(DEP)/stb -I $(DEP)/tinyobj
LFLAGS		:= -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

RM			:= rm -rf

DEBUG		?= 0

all: $(NAME)

$(NAME): $(OBJS)
	glslc shaders/shader.frag -o shaders/frag.spv
	glslc shaders/shader.vert -o shaders/vert.spv
	$(CC) $(CFLAGS) $(OBJS) $(LFLAGS) -o $(NAME)

$(DIR_OBJS)/%.o: $(DIR_SRCS)/%.cpp
	mkdir -p $(DIR_OBJS)
	$(CC) $(CFLAGS) $(IFLAGS) -D DEBUG=$(DEBUG) -o $@ -c $<

shaders:
	glslc shaders/shader.frag -o shaders/frag.spv
	glslc shaders/shader.vert -o shaders/vert.spv

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)
	glslc shaders/shader.frag -o shaders/frag.spv
	glslc shaders/shader.vert -o shaders/vert.spv
	rm shaders/frag.spv
	rm shaders/vert.spv

re: fclean all

.PHONY :all shaders clean fclean re
