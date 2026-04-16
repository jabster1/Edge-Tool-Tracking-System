/*
 * benchmark_cpp.cpp
 * ONNX Runtime C++ inference benchmark for FOD Tool Tracking System.
 *
 * Usage:
 *   ./benchmark_cpp        → CPU provider  (saves benchmark_results_cpp_cpu.json)
 *   ./benchmark_cpp coreml → CoreML provider (saves benchmark_results_cpp_coreml.json)
 *
 * Both variants open webcam device 0 (cv::VideoCapture(0)), run 100 frames,
 * draw bounding boxes + HUD on a live window, and record avg/min/max timing.
 *
 * Platform : macOS Apple Silicon (arm64)
 * Build    : see CMakeLists.txt
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>   // OrtSessionOptionsAppendExecutionProvider_CoreML

// ── Constants ─────────────────────────────────────────────────────────────────

static const int   NUM_FRAMES      = 100;
static const int   INPUT_W         = 512;
static const int   INPUT_H         = 512;
static const float CONF_THRESHOLD  = 0.50f;
static const float IOU_THRESHOLD   = 0.45f;

static const std::array<std::string, 5> CLASS_NAMES = {
    "drill", "hammer", "pliers", "screwdriver", "wrench"
};

// BGR palette – matches Python benchmark colours
static const std::array<cv::Scalar, 5> COLOURS = {{
    {  0, 165, 255},   // drill        – orange
    {  0, 255,   0},   // hammer       – green
    {255,   0,   0},   // pliers       – blue
    {  0,   0, 255},   // screwdriver  – red
    {255,   0, 255},   // wrench       – magenta
}};

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string doubleToStr(double v, int dec = 4) {
    std::ostringstream ss;
    ss << std::fixed;
    ss.precision(dec);
    ss << v;
    return ss.str();
}

// ── Preprocessing ─────────────────────────────────────────────────────────────
// Resize to INPUT_W×INPUT_H, BGR→RGB, normalise [0,1], HWC→CHW.

static std::vector<float> preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);

    std::vector<float> blob;
    blob.reserve(3 * INPUT_H * INPUT_W);
    for (int c = 0; c < 3; ++c) {
        const float* p = reinterpret_cast<const float*>(ch[c].data);
        blob.insert(blob.end(), p, p + INPUT_H * INPUT_W);
    }
    return blob;
}

// ── Detection + NMS ──────────────────────────────────────────────────────────

struct Det { float x1, y1, x2, y2, conf; int cls; };

static float iou(const Det& a, const Det& b) {
    float ix1  = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2  = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float ua    = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / (ua + 1e-6f);
}

static std::vector<Det> nms(std::vector<Det> dets) {
    std::sort(dets.begin(), dets.end(),
              [](const Det& a, const Det& b){ return a.conf > b.conf; });
    std::vector<bool> sup(dets.size(), false);
    std::vector<Det>  out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!sup[j] && dets[i].cls == dets[j].cls
                && iou(dets[i], dets[j]) > IOU_THRESHOLD)
                sup[j] = true;
    }
    return out;
}

// YOLOv8/11 output layout: [1, 4+nc, num_anchors]
static std::vector<Det> postprocess(const float* data,
                                    int num_anchors, int num_classes,
                                    int orig_w, int orig_h) {
    float sx = static_cast<float>(orig_w) / INPUT_W;
    float sy = static_cast<float>(orig_h) / INPUT_H;

    std::vector<Det> raw;
    for (int a = 0; a < num_anchors; ++a) {
        float cx = data[0 * num_anchors + a];
        float cy = data[1 * num_anchors + a];
        float bw = data[2 * num_anchors + a];
        float bh = data[3 * num_anchors + a];

        float best_score = 0.f; int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = data[(4 + c) * num_anchors + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < CONF_THRESHOLD) continue;

        float x1 = std::max(0.f, (cx - bw * 0.5f) * sx);
        float y1 = std::max(0.f, (cy - bh * 0.5f) * sy);
        float x2 = std::min((float)orig_w, (cx + bw * 0.5f) * sx);
        float y2 = std::min((float)orig_h, (cy + bh * 0.5f) * sy);
        raw.push_back({x1, y1, x2, y2, best_score, best_cls});
    }
    return nms(raw);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // ── Provider selection ────────────────────────────────────────────────
    bool use_coreml = (argc > 1 && std::string(argv[1]) == "coreml");

    const std::string provider_label = use_coreml ? "CoreML" : "CPU";
    const std::string output_json    = use_coreml
        ? "benchmark_results_cpp_coreml.json"
        : "benchmark_results_cpp_cpu.json";
    const std::string window_name    =
        "Benchmark — C++ ONNX Runtime [" + provider_label + "] (press q to quit early)";

    std::cout << "=== C++ ONNX Runtime Benchmark — provider: "
              << provider_label << " ===\n\n";

    // ── ONNX Runtime session ──────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_coreml) {
        // COREML_FLAG_CREATE_MLPROGRAM: use MLProgram format (better ANE
        // utilisation on Apple Silicon vs legacy NeuralNetwork format).
        // Falls back to CPU for any unsupported ops automatically.
        uint32_t coreml_flags = COREML_FLAG_CREATE_MLPROGRAM;
        auto status = OrtSessionOptionsAppendExecutionProvider_CoreML(
            opts, coreml_flags);
        if (status != nullptr) {
            std::cerr << "WARNING: CoreML EP could not be registered: "
                      << Ort::GetApi().GetErrorMessage(status) << "\n"
                      << "Falling back to CPU.\n";
            Ort::GetApi().ReleaseStatus(status);
        } else {
            std::cout << "CoreML provider registered "
                      << "(MLProgram, ANE+GPU+CPU compute units).\n";
        }
        // CoreML is best with multiple CPU threads for the fallback ops
        opts.SetIntraOpNumThreads(4);
    } else {
        // Single thread to match the Python benchmark's single-forward-pass cost
        opts.SetIntraOpNumThreads(1);
    }

    std::cout << "Loading model: best.onnx\n";
    Ort::Session session(env, "best.onnx", opts);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr  = session.GetInputNameAllocated(0, alloc);
    auto out_name_ptr = session.GetOutputNameAllocated(0, alloc);
    std::string in_name  = in_name_ptr.get();
    std::string out_name = out_name_ptr.get();
    std::cout << "  Input  : " << in_name  << "\n";
    std::cout << "  Output : " << out_name << "\n";

    auto out_shape  = session.GetOutputTypeInfo(0)
                             .GetTensorTypeAndShapeInfo().GetShape();
    int num_classes = (out_shape.size() >= 2 && out_shape[1] > 4)
                    ? static_cast<int>(out_shape[1]) - 4
                    : static_cast<int>(CLASS_NAMES.size());

    std::array<int64_t, 4> input_shape{1, 3, INPUT_H, INPUT_W};
    const size_t input_numel = 3 * INPUT_H * INPUT_W;

    const char* in_names[]  = {in_name.c_str()};
    const char* out_names[] = {out_name.c_str()};

    // ── Webcam — device 0, identical to Python benchmark ─────────────────
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam (device 0).\n"
                  << "Check System Settings → Privacy & Security → Camera.\n";
        return 1;
    }

    // macOS AVFoundation warmup — drain frames until pixels arrive
    std::cout << "\nWarming up camera ...\n";
    for (int w = 0; w < 10; ++w) {
        cv::Mat tmp; cap.read(tmp);
        if (!tmp.empty()) break;
        cv::waitKey(100);
    }

    cv::Mat test_frame;
    if (!cap.read(test_frame) || test_frame.empty()) {
        std::cerr << "ERROR: Camera opened but returned no frames.\n";
        cap.release();
        return 1;
    }
    std::cout << "Camera ready — "
              << test_frame.cols << "x" << test_frame.rows << "\n";

    // ── CoreML model compilation warm-up ─────────────────────────────────
    // The first inference with CoreML triggers on-device model compilation
    // which can take several seconds. Run one throwaway frame so the timed
    // loop measures steady-state latency only.
    if (use_coreml) {
        std::cout << "Compiling CoreML model (first-run only, may take a few seconds) ...\n";
        std::vector<float> warmup_blob = preprocess(test_frame);
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value wt = Ort::Value::CreateTensor<float>(
            mi, warmup_blob.data(), input_numel,
            input_shape.data(), input_shape.size());
        try {
            session.Run(Ort::RunOptions{nullptr}, in_names, &wt, 1, out_names, 1);
            std::cout << "CoreML model compiled OK.\n";
        } catch (const Ort::Exception& e) {
            std::cerr << "CoreML warmup failed: " << e.what() << "\n";
        }
    }

    std::cout << "\nRunning benchmark — " << NUM_FRAMES << " frames at "
              << INPUT_W << "x" << INPUT_H << "\n";
    std::cout << "Close the window or press 'q' to stop early.\n\n";

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // ── Benchmark loop ────────────────────────────────────────────────────
    std::vector<double> frame_times;
    frame_times.reserve(NUM_FRAMES);
    int frames_captured = 0;

    for (int i = 0; i < NUM_FRAMES; ++i) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Warning: failed to read frame " << i
                      << ", stopping early\n";
            break;
        }
        ++frames_captured;

        std::vector<float> blob = preprocess(frame);

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, blob.data(), input_numel,
            input_shape.data(), input_shape.size());

        // ── Timed inference ───────────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, 1);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_times.push_back(elapsed_ms);

        // ── Decode + draw detections ──────────────────────────────────────
        const float* out_data = outputs[0].GetTensorData<float>();
        auto out_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_anchors = (out_dims.size() >= 3)
                        ? static_cast<int>(out_dims[2]) : 0;

        std::vector<Det> dets = postprocess(
            out_data, num_anchors, num_classes, frame.cols, frame.rows);

        for (const auto& d : dets) {
            cv::Scalar col = COLOURS[d.cls % COLOURS.size()];
            std::string name  = (d.cls < (int)CLASS_NAMES.size())
                                 ? CLASS_NAMES[d.cls]
                                 : "cls" + std::to_string(d.cls);
            std::string label = name + " " +
                                std::to_string((int)(d.conf * 100)) + "%";
            cv::rectangle(frame,
                cv::Point((int)d.x1, (int)d.y1),
                cv::Point((int)d.x2, (int)d.y2), col, 2);
            cv::putText(frame, label,
                cv::Point((int)d.x1, (int)d.y1 - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2);
        }

        // ── HUD ───────────────────────────────────────────────────────────
        double avg_so_far = std::accumulate(frame_times.begin(),
                                            frame_times.end(), 0.0)
                          / (double)frame_times.size();
        char hud[160];
        std::snprintf(hud, sizeof(hud),
            "[%s] Frame %d/%d  |  This: %.1f ms  |  Avg: %.1f ms  |  FPS: %.1f",
            provider_label.c_str(), i + 1, NUM_FRAMES,
            elapsed_ms, avg_so_far, 1000.0 / avg_so_far);
        cv::putText(frame, hud, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.60,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        cv::imshow(window_name, frame);

        if ((i + 1) % 10 == 0)
            std::printf("  Frame %3d/%d  %7.2f ms  (avg %.2f ms)\n",
                        i + 1, NUM_FRAMES, elapsed_ms, avg_so_far);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            std::cout << "Early exit requested.\n";
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    if (frame_times.empty()) {
        std::cerr << "ERROR: No frames were processed.\n";
        return 1;
    }

    // ── Statistics ────────────────────────────────────────────────────────
    double sum   = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
    double avg   = sum / (double)frame_times.size();
    double fps   = 1000.0 / avg;
    double min_t = *std::min_element(frame_times.begin(), frame_times.end());
    double max_t = *std::max_element(frame_times.begin(), frame_times.end());

    std::printf("\n=== C++ ONNX Runtime [%s] Benchmark Results (%d frames) ===\n",
                provider_label.c_str(), frames_captured);
    std::printf("  Avg inference time : %.2f ms\n", avg);
    std::printf("  FPS                : %.2f\n",    fps);
    std::printf("  Min                : %.2f ms\n", min_t);
    std::printf("  Max                : %.2f ms\n", max_t);

    // ── Save JSON ─────────────────────────────────────────────────────────
    std::ofstream jf(output_json);
    if (!jf.is_open()) {
        std::cerr << "ERROR: Could not write " << output_json << "\n";
        return 1;
    }
    std::string backend = "C++ ONNX Runtime (" + provider_label + ")";
    jf << "{\n"
       << "  \"backend\": \""    << backend          << "\",\n"
       << "  \"model\": \"best.onnx\",\n"
       << "  \"provider\": \""   << provider_label   << "\",\n"
       << "  \"imgsz\": "        << INPUT_W          << ",\n"
       << "  \"frames\": "       << frames_captured  << ",\n"
       << "  \"avg_inference_ms\": " << doubleToStr(avg)   << ",\n"
       << "  \"fps\": "              << doubleToStr(fps)   << ",\n"
       << "  \"min_inference_ms\": " << doubleToStr(min_t) << ",\n"
       << "  \"max_inference_ms\": " << doubleToStr(max_t) << "\n"
       << "}\n";
    jf.close();

    std::cout << "\nResults saved to " << output_json << "\n";
    return 0;
}
