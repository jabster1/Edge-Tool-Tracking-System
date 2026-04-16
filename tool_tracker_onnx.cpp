/*
 * tool_tracker_onnx.cpp
 * Real-time FOD Tool Tracking System using ONNX Runtime + OpenCV.
 *
 * Loads best.onnx, runs inference on each webcam frame, tracks
 * presence / absence of the five industrial tools, and logs
 * timestamped events to tool_log.txt.
 *
 * Classes (index → name):
 *   0 drill | 1 hammer | 2 pliers | 3 screwdriver | 4 wrench
 *
 * Controls:
 *   q  – quit
 *   c  – calibrate (register currently-visible tools as the baseline)
 *
 * Platform: macOS Apple Silicon
 * Build   : see CMakeLists.txt
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ── Configuration ─────────────────────────────────────────────────────────────

static const std::string MODEL_PATH   = "best.onnx";
static const std::string LOG_FILE     = "tool_log.txt";
static const int   INPUT_W            = 512;
static const int   INPUT_H            = 512;
static const float CONF_THRESHOLD     = 0.50f;
static const float IOU_THRESHOLD      = 0.45f;
static const int   LOG_INTERVAL_SEC   = 5;   // min seconds between repeated logs per class

static const std::array<std::string, 5> CLASS_NAMES = {
    "drill", "hammer", "pliers", "screwdriver", "wrench"
};

// ── Utility ───────────────────────────────────────────────────────────────────

static std::string timestamp_now() {
    auto now   = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static void log_event(std::ofstream& log, const std::string& msg) {
    std::cout << msg << "\n";
    log << msg << "\n";
    log.flush();
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

static std::vector<float> preprocess(const cv::Mat& frame,
                                     float& scale,
                                     int& pad_left, int& pad_top) {
    // Letterbox resize: keep aspect ratio, pad with grey (114)
    float r = std::min(static_cast<float>(INPUT_W) / frame.cols,
                       static_cast<float>(INPUT_H) / frame.rows);
    scale = r;

    int new_w = static_cast<int>(std::round(frame.cols * r));
    int new_h = static_cast<int>(std::round(frame.rows * r));
    pad_left  = (INPUT_W - new_w) / 2;
    pad_top   = (INPUT_H - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_left, pad_top, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    // HWC → CHW
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    std::vector<float> blob;
    blob.reserve(3 * INPUT_H * INPUT_W);
    for (int c = 0; c < 3; ++c) {
        const float* p = reinterpret_cast<const float*>(channels[c].data);
        blob.insert(blob.end(), p, p + INPUT_H * INPUT_W);
    }
    return blob;
}

// ── NMS ───────────────────────────────────────────────────────────────────────

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int   cls;
};

static float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.conf > b.conf; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> out;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].cls == dets[j].cls &&
                iou(dets[i], dets[j]) > iou_thresh)
                suppressed[j] = true;
        }
    }
    return out;
}

// ── Post-processing ───────────────────────────────────────────────────────────
// YOLOv8/11 output: [1, 4+nc, num_anchors] where rows are (x_c, y_c, w, h, cls0..clsN)

static std::vector<Detection> postprocess(const float* data,
                                          int num_anchors,
                                          int num_classes,
                                          float conf_thresh,
                                          float scale,
                                          int pad_left, int pad_top,
                                          int orig_w,  int orig_h) {
    std::vector<Detection> raw;
    // data layout: [4 + num_classes, num_anchors], column-major iteration
    for (int a = 0; a < num_anchors; ++a) {
        float cx = data[0 * num_anchors + a];
        float cy = data[1 * num_anchors + a];
        float bw = data[2 * num_anchors + a];
        float bh = data[3 * num_anchors + a];

        // Find best class
        float best_score = 0.0f;
        int   best_cls   = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = data[(4 + c) * num_anchors + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thresh) continue;

        // Convert to original image coordinates
        float x1 = (cx - bw / 2.0f - pad_left) / scale;
        float y1 = (cy - bh / 2.0f - pad_top)  / scale;
        float x2 = (cx + bw / 2.0f - pad_left) / scale;
        float y2 = (cy + bh / 2.0f - pad_top)  / scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        raw.push_back({x1, y1, x2, y2, best_score, best_cls});
    }
    return nms(raw, IOU_THRESHOLD);
}

// ── Colour palette (BGR) ──────────────────────────────────────────────────────

static cv::Scalar class_colour(int cls) {
    static const std::array<cv::Scalar, 5> palette = {{
        {  0, 165, 255},   // drill        – orange
        {  0, 255,   0},   // hammer       – green
        {255,   0,   0},   // pliers       – blue
        {  0,   0, 255},   // screwdriver  – red
        {255,   0, 255},   // wrench       – magenta
    }};
    return palette[cls % palette.size()];
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // ── ONNX Runtime ─────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "tool_tracker");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading " << MODEL_PATH << " ...\n";
    Ort::Session session(env, MODEL_PATH.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr  = session.GetInputNameAllocated(0, alloc);
    auto out_name_ptr = session.GetOutputNameAllocated(0, alloc);
    std::string in_name  = in_name_ptr.get();
    std::string out_name = out_name_ptr.get();

    // Infer number of classes from output shape
    auto out_shape = session.GetOutputTypeInfo(0)
                            .GetTensorTypeAndShapeInfo()
                            .GetShape();
    // shape: [1, 4+nc, anchors]  → out_shape[1] - 4 = nc
    int num_classes = (out_shape.size() >= 2 && out_shape[1] > 4)
                    ? static_cast<int>(out_shape[1]) - 4
                    : static_cast<int>(CLASS_NAMES.size());
    std::cout << "Model reports " << num_classes << " classes, "
              << "using " << CLASS_NAMES.size() << " known names.\n";

    std::array<int64_t, 4> input_shape{1, 3, INPUT_H, INPUT_W};
    size_t input_numel = 3 * INPUT_H * INPUT_W;

    // ── Log file ─────────────────────────────────────────────────────────
    std::ofstream log(LOG_FILE, std::ios::app);
    if (!log.is_open()) {
        std::cerr << "ERROR: Cannot open " << LOG_FILE << "\n";
        return 1;
    }
    log_event(log, "=== Tool Tracker (ONNX Runtime) started at " + timestamp_now() + " ===");

    // ── Webcam ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open webcam (device 0)\n";
        return 1;
    }

    // ── State ────────────────────────────────────────────────────────────
    // calibrated_tools: set of class IDs that were present at calibration time
    std::set<int> calibrated_tools;
    bool calibrated = false;

    // last_seen[cls] = last time the tool was seen this session (for debounce)
    std::map<int, std::chrono::steady_clock::time_point> last_logged;

    // present_state[cls]: true if tool was present on the previous frame
    std::map<int, bool> present_state;

    auto can_log = [&](int cls) -> bool {
        auto it = last_logged.find(cls);
        if (it == last_logged.end()) return true;
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - it->second).count();
        return elapsed >= LOG_INTERVAL_SEC;
    };

    std::cout << "\nStarting real-time tool tracking...\n";
    std::cout << "  Press 'c' to calibrate (set current tools as baseline)\n";
    std::cout << "  Press 'q' to quit\n\n";

    const char* in_names[]  = {in_name.c_str()};
    const char* out_names[] = {out_name.c_str()};

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        int orig_w = frame.cols, orig_h = frame.rows;

        // ── Preprocess ───────────────────────────────────────────────────
        float scale;
        int pad_left, pad_top;
        std::vector<float> blob = preprocess(frame, scale, pad_left, pad_top);

        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, blob.data(), input_numel,
            input_shape.data(), input_shape.size());

        // ── Inference ────────────────────────────────────────────────────
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            in_names, &input_tensor, 1,
            out_names, 1);

        const float* out_data = outputs[0].GetTensorData<float>();
        auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
        auto out_dims = out_info.GetShape();
        // Expected: [1, 4+nc, num_anchors]
        int num_anchors = (out_dims.size() >= 3) ? static_cast<int>(out_dims[2]) : 0;

        std::vector<Detection> dets = postprocess(
            out_data, num_anchors, num_classes, CONF_THRESHOLD,
            scale, pad_left, pad_top, orig_w, orig_h);

        // ── Which classes are visible this frame ─────────────────────────
        std::set<int> visible_classes;
        for (const auto& d : dets)
            visible_classes.insert(d.cls);

        std::string ts = timestamp_now();

        // ── Presence / absence tracking ──────────────────────────────────
        if (calibrated) {
            for (int cls : calibrated_tools) {
                bool now_present  = visible_classes.count(cls) > 0;
                bool was_present  = present_state.count(cls) && present_state[cls];

                if (!now_present && was_present && can_log(cls)) {
                    std::string name = (cls < static_cast<int>(CLASS_NAMES.size()))
                                       ? CLASS_NAMES[cls] : "class_" + std::to_string(cls);
                    log_event(log, "[" + ts + "] MISSING: " + name);
                    last_logged[cls] = std::chrono::steady_clock::now();
                }
                if (now_present && !was_present && can_log(cls)) {
                    std::string name = (cls < static_cast<int>(CLASS_NAMES.size()))
                                       ? CLASS_NAMES[cls] : "class_" + std::to_string(cls);
                    log_event(log, "[" + ts + "] RETURNED: " + name);
                    last_logged[cls] = std::chrono::steady_clock::now();
                }
                present_state[cls] = now_present;
            }
        } else {
            // Before calibration: just log first detection of each class
            for (int cls : visible_classes) {
                if (!present_state.count(cls) && can_log(cls)) {
                    std::string name = (cls < static_cast<int>(CLASS_NAMES.size()))
                                       ? CLASS_NAMES[cls] : "class_" + std::to_string(cls);
                    log_event(log, "[" + ts + "] DETECTED: " + name + " (pre-calibration)");
                    last_logged[cls] = std::chrono::steady_clock::now();
                }
                present_state[cls] = true;
            }
        }

        // ── Draw detections ───────────────────────────────────────────────
        for (const auto& d : dets) {
            cv::Scalar colour = class_colour(d.cls);
            std::string name  = (d.cls < static_cast<int>(CLASS_NAMES.size()))
                                 ? CLASS_NAMES[d.cls] : "?";
            std::string label = name + " " +
                                std::to_string(static_cast<int>(d.conf * 100)) + "%";

            cv::rectangle(frame,
                          cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                          cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
                          colour, 2);
            cv::putText(frame, label,
                        cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1) - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, colour, 2);
        }

        // ── HUD ──────────────────────────────────────────────────────────
        std::string hud_mode = calibrated ? "MONITORING" : "PRE-CALIBRATION";
        cv::putText(frame, hud_mode,
                    cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    calibrated ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 2);

        int tool_y = 60;
        for (int cls : calibrated_tools) {
            bool present = visible_classes.count(cls) > 0;
            std::string name = (cls < static_cast<int>(CLASS_NAMES.size()))
                               ? CLASS_NAMES[cls] : "class_" + std::to_string(cls);
            std::string status = present ? "[OK]    " : "[MISSING]";
            cv::Scalar col = present ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(frame, status + " " + name,
                        cv::Point(10, tool_y), cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2);
            tool_y += 24;
        }

        cv::imshow("FOD Tool Tracker (ONNX Runtime)", frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        if (key == 'c') {
            calibrated_tools = visible_classes;
            calibrated = true;
            // Reset presence state
            present_state.clear();
            for (int cls : calibrated_tools) present_state[cls] = true;

            std::ostringstream cal_msg;
            cal_msg << "[" << ts << "] === CALIBRATION: registered "
                    << calibrated_tools.size() << " tools: ";
            for (int cls : calibrated_tools) {
                if (cls < static_cast<int>(CLASS_NAMES.size()))
                    cal_msg << CLASS_NAMES[cls] << " ";
            }
            cal_msg << "===";
            log_event(log, cal_msg.str());
        }
    }

    cap.release();
    cv::destroyAllWindows();

    log_event(log, "=== Tool Tracker stopped at " + timestamp_now() + " ===\n");
    log.close();

    std::cout << "Stopped. Log saved to " << LOG_FILE << "\n";
    return 0;
}
