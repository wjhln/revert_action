#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <nlopt.hpp>
#include <cppad/cppad.hpp>

namespace vehicle {

template<typename T = double>
struct State {
    T x;      // 位置x
    T y;      // 位置y
    T psi;    // 航向角
    T v;      // 速度

    State(T x = T(0), T y = T(0), T psi = T(0), T v = T(0)) 
        : x(x), y(y), psi(psi), v(v) {}
};

template<typename T = double>
struct Control {
    T a;      // 加速度
    T delta;  // 转向角
    
    Control(T a = T(0), T delta = T(0)) 
        : a(a), delta(delta) {}
};

struct TrajectoryData {
    std::vector<double> x, y;           // 位置数据
    std::vector<double> heading;        // 航向角数据
    std::vector<double> velocity;       // 速度数据
    std::vector<double> acceleration;   // 加速度数据
    double dt;                          // 时间步长
};

class TrajectoryOptimizer {
private:
    static constexpr double L = 3.0;  // 车辆轴距
    static constexpr double Q_pos = 100.0;     // 位置误差权重
    static constexpr double Q_heading = 0.0;   // 航向误差权重
    
    struct OptData {
        TrajectoryData* trajData;
        State<double> x0;
        double dt;
        TrajectoryOptimizer* optimizer;
    };

    using ADvector = CPPAD_TESTVECTOR(CppAD::AD<double>);
    
public:
    TrajectoryData loadTrajectory(const std::string& filepath) {
        TrajectoryData data;
        std::ifstream file(filepath);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filepath << std::endl;
            return data;
        }
        
        std::getline(file, line);
        
        std::vector<double> timestamps;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            
            if (row.size() >= 6) {
                timestamps.push_back(std::stod(row[0]));
                data.x.push_back(std::stod(row[1]));
                data.y.push_back(std::stod(row[2]));
                data.heading.push_back(std::stod(row[3]));
                data.velocity.push_back(std::stod(row[4]));
                data.acceleration.push_back(std::stod(row[5]));
            }
        }
        
        data.dt = timestamps.size() > 1 ? timestamps[1] - timestamps[0] : 0.1;
        return data;
    }
    
    template<typename T>
    std::vector<State<T>> predict(const State<T>& x0, 
                                const std::vector<Control<T>>& controls, 
                                const T& dt) {
        std::vector<State<T>> states;
        states.reserve(controls.size() + 1);
        states.push_back(x0);
        
        for (const auto& u : controls) {
            const State<T>& s = states.back();
            T x_next, y_next, psi_next, v_next;
            
            if constexpr (std::is_same_v<T, CppAD::AD<double>>) {
                x_next = s.x + s.v * CppAD::cos(s.psi) * dt;
                y_next = s.y + s.v * CppAD::sin(s.psi) * dt;
                psi_next = s.psi + (s.v / T(L)) * CppAD::tan(u.delta) * dt;
                v_next = CppAD::CondExpGt(s.v + u.a * dt, T(0), s.v + u.a * dt, T(0));
            } else {
                x_next = s.x + s.v * std::cos(s.psi) * dt;
                y_next = s.y + s.v * std::sin(s.psi) * dt;
                psi_next = s.psi + (s.v / L) * std::tan(u.delta) * dt;
                v_next = std::max(T(0), s.v + u.a * dt);
            }
            
            states.emplace_back(x_next, y_next, psi_next, v_next);
        }
        
        return states;
    }

    template<typename T>
    T calculateCost(const std::vector<Control<T>>& controls, 
                   const State<T>& x0,
                   const TrajectoryData& trajData, 
                   const T& dt) {
        auto states = predict(x0, controls, dt);
        T cost = T(0.0);
        
        int minLen = std::min(states.size(), trajData.x.size());
        for (int i = 0; i < minLen; i++) {
            T dx = states[i].x - T(trajData.x[i]);
            T dy = states[i].y - T(trajData.y[i]);
            
            T dpsi = T(0.0);
            if (i < static_cast<int>(trajData.heading.size())) {
                dpsi = states[i].psi - T(trajData.heading[i]);
            }
            
            cost += T(Q_pos) * (dx * dx + dy * dy) + T(Q_heading) * dpsi * dpsi;
        }
        
        return cost;
    }

    void calculateGradient(const std::vector<double>& x, std::vector<double>& grad,
                         const State<double>& x0, const TrajectoryData& trajData, double dt) {
        ADvector ax(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            ax[i] = x[i];
        }
        
        CppAD::Independent(ax);
        
        int N = x.size() / 2;
        std::vector<Control<CppAD::AD<double>>> controls(N);
        for (int i = 0; i < N; i++) {
            controls[i].a = ax[2*i];
            controls[i].delta = ax[2*i + 1];
        }
        
        State<CppAD::AD<double>> x0_ad(
            CppAD::AD<double>(x0.x),
            CppAD::AD<double>(x0.y),
            CppAD::AD<double>(x0.psi),
            CppAD::AD<double>(x0.v)
        );
        
        ADvector ay(1);
        ay[0] = calculateCost(controls, x0_ad, trajData, CppAD::AD<double>(dt));
        
        CppAD::ADFun<double> f(ax, ay);
        grad = f.Jacobian(x);
    }
    
    static double objectiveFunc(const std::vector<double>& x, std::vector<double>& grad, void* data) {
        OptData* optData = static_cast<OptData*>(data);
        int N = x.size() / 2;
        
        std::vector<Control<double>> controls(N);
        for (int i = 0; i < N; i++) {
            controls[i].a = x[2*i];
            controls[i].delta = x[2*i + 1];
        }
        
        double cost = optData->optimizer->calculateCost(
            controls, optData->x0, *optData->trajData, optData->dt);
        
        if (!grad.empty()) {
            optData->optimizer->calculateGradient(
                x, grad, optData->x0, *optData->trajData, optData->dt);
        }
        
        return cost;
    }
    
    std::vector<State<double>> optimize(const TrajectoryData& trajData, int N) {
        State<double> x0(trajData.x[0], trajData.y[0], trajData.heading[0], trajData.velocity[0]);
        
        nlopt::opt opt(nlopt::LD_SLSQP, 2 * N);
        OptData optData{const_cast<TrajectoryData*>(&trajData), x0, trajData.dt, this};
        opt.set_min_objective(objectiveFunc, &optData);
        
        std::vector<double> lb(2 * N, -4.0), ub(2 * N, 4.0);
        for (int i = 1; i < 2 * N; i += 2) {
            lb[i] = -2.5;
            ub[i] = 2.5;
        }
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        
        opt.set_ftol_rel(1e-6);
        opt.set_maxeval(300);
        
        std::vector<double> x(2 * N, 0.0);
        for (int k = 0; k < N; k++) {
            if (k < static_cast<int>(trajData.acceleration.size())) {
                x[2*k] = trajData.acceleration[k];
            }
            
            if (k < static_cast<int>(trajData.heading.size()) - 2) {
                double dpsi = (trajData.heading[k+2] - trajData.heading[k]) / 2.0;
                double v_k = (k < static_cast<int>(trajData.velocity.size())) ? 
                           std::max(trajData.velocity[k], 0.1) : 1.0;
                x[2*k + 1] = std::atan(dpsi * L / (v_k * trajData.dt));
                x[2*k + 1] = std::clamp(x[2*k + 1], -2.5, 2.5);
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        double minCost;
        try {
            opt.optimize(x, minCost);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            
            std::cout << "最终代价: " << minCost << std::endl;
            std::cout << "求解耗时: " << duration.count() << " 毫秒 (" 
                      << duration.count() / 1000.0 << " 秒)" << std::endl;
        } catch(std::exception &e) {
            std::cerr << "优化异常: " << e.what() << std::endl;
        }
        
        std::vector<Control<double>> optControls(N);
        for (int i = 0; i < N; i++) {
            optControls[i].a = x[2*i];
            optControls[i].delta = x[2*i + 1];
        }
        
        return predict(x0, optControls, trajData.dt);
    }
    
    void saveResults(const TrajectoryData& original, const std::vector<State<double>>& optimized) {
        std::ofstream file("trajectory_results.csv");
        if (!file.is_open()) return;
        
        file << "orig_x,orig_y,opt_x,opt_y\n";
        
        int minSize = std::min(original.x.size(), optimized.size());
        for (int i = 0; i < minSize; i++) {
            file << original.x[i] << "," << original.y[i] << ","
                 << optimized[i].x << "," << optimized[i].y << "\n";
        }
    }
};

} // namespace vehicle

int main() {
    vehicle::TrajectoryOptimizer optimizer;
    
    vehicle::TrajectoryData trajData = optimizer.loadTrajectory("/home/wang/Project/Sampling/revert_action/trajectory_dataset/trajectory_0005.csv");
    if (trajData.x.empty()) {
        std::cerr << "轨迹数据加载失败！" << std::endl;
        return -1;
    }
    
    int N = static_cast<int>(trajData.x.size()) - 1;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    auto optimizedTrajectory = optimizer.optimize(trajData, N);
    auto total_end = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    std::cout << "总执行时间: " << total_duration.count() << " 毫秒 (" 
              << total_duration.count() / 1000.0 << " 秒)" << std::endl;
    
    optimizer.saveResults(trajData, optimizedTrajectory);
    
    return 0;
} 