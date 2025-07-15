NAME		:= Vulkan

SRCS		:= srcs/App.cpp

DIR_SRCS	:= srcs
DIR_OBJS	:= .objs

OBJS		:= $(addprefix $(DIR_OBJS)/, $(notdir $(SRCS:.cpp=.o)))

CC			:= c++
CFLAGS		:= -Wall -Werror -Wextra -std=c++17 -g
IFLAGS		:= -I include
LFLAGS		:= -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

RM			:= rm -rf

DEBUG		?= 0

all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LFLAGS) -o $(NAME)

$(DIR_OBJS)/%.o: $(DIR_SRCS)/%.cpp
	mkdir -p $(DIR_OBJS)
	$(CC) $(CFLAGS) $(IFLAGS) -D DEBUG=$(DEBUG) -o $@ -c $<

clean:
	$(RM) $(OBJS)

fclean: clean
	$(RM) $(NAME)

re: fclean all
