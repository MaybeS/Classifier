#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include <chrono>

template<typename T = std::chrono::milliseconds, size_t N = 10>
struct measure {
    template<typename F, typename ...Args>
    static auto execution(F&& f, Args&&... args) {
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < N; i++) {
            std::forward<decltype(f)>(f)(std::forward<Args>(args)...);
        }

        return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - start);
    }

    static auto logging(const T& duration, const std::string& text) {
        std::cout << text << " (" << N << " times)" << std::endl;
        std::cout << "\tTotal: " << duration.count() << " ms" << std::endl;
        std::cout << "\tMean: " << duration.count() / (double)N << " ms" << std::endl;
    }
};

const std::string IMAGE{ "/home/jiun/workspace/Classifier/libtorch/bin/test.jpg" };
const std::string MODULE_NAME{ "/home/jiun/workspace/Classifier/libtorch/bin/amano-mb2-128.pt" };
const cv::Size size{ 128, 128 };

int main() {
    torch::jit::script::Module module;
    torch::Tensor image_tensor;
    std::vector<torch::jit::IValue> inputs;
    cv::Mat image;
    int results = 0;


    measure<>::logging(measure<>::execution([&image]() {
        image = cv::imread(IMAGE);
    }), "Image read");

    measure<>::logging(measure<>::execution([&module]() {
        module = torch::jit::load(MODULE_NAME);
        module.eval();
    }), "Module Init");

    module.eval();
    cv::Mat dst;
    cv::resize(image, dst, size);
    dst.convertTo(dst, CV_32FC3, 1 / 255.f);

    std::vector<int64_t> shape{ 1,
                                static_cast<int64_t>(dst.rows), static_cast<int64_t>(dst.cols), static_cast<int64_t>(dst.channels()) };
    torch::TensorOptions options(torch::kFloat);
    image_tensor = torch::from_blob(dst.data, torch::IntList(shape), options).permute({ 0, 3, 1, 2 });
    image_tensor.requires_grad_(false);

    inputs = { image_tensor };

    measure<>::logging(measure<>::execution([&module, &results, &inputs]() {
        auto output = module.forward(inputs).toTensor();
        // results = torch::argmax(output, 1).item<int>();
    }), "Module Detection");
}
