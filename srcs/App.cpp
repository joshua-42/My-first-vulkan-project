#include <App.hpp>

int main(int argc, char **argv) {
    App app;

    if (argc != 2) {
        std::cout << "Please give an .obj file\n";
        return 1;
    }
    try {
        MODEL_PATH = argv[1];
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
