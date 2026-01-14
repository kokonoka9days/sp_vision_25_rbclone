// debug_yaml.cpp
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <filesystem>

void checkYAML(const std::string& filename) {
    std::cout << "检查YAML文件: " << filename << std::endl;
    
    // 检查文件是否存在
    if (!std::filesystem::exists(filename)) {
        std::cerr << "错误：文件不存在" << std::endl;
        return;
    }
    
    // 读取原始文件内容
    std::ifstream file(filename);
    std::string content((std::istreambuf_iterator<char>(file)), 
                       std::istreambuf_iterator<char>());
    file.close();
    
    std::cout << "文件大小: " << content.size() << " 字节" << std::endl;
    
    try {
        // 尝试解析YAML
        YAML::Node yaml = YAML::LoadFile(filename);
        std::cout << "✅ YAML文件解析成功" << std::endl;
        
        // 尝试读取一些关键字段
        if (yaml["yolo_name"]) {
            std::cout << "✅ yolo_name: " << yaml["yolo_name"].as<std::string>() << std::endl;
        }
        
        if (yaml["device"]) {
            std::cout << "✅ device: " << yaml["device"].as<std::string>() << std::endl;
        }
        
        if (yaml["camera_name"]) {
            std::cout << "✅ camera_name: " << yaml["camera_name"].as<std::string>() << std::endl;
        }
        
        // 检查数组类型的字段
        if (yaml["camera_matrix"]) {
            std::cout << "✅ camera_matrix: 数组类型，大小: " << yaml["camera_matrix"].size() << std::endl;
            try {
                auto matrix = yaml["camera_matrix"].as<std::vector<double>>();
                std::cout << "  数组值: ";
                for (size_t i = 0; i < std::min(matrix.size(), size_t(5)); ++i) {
                    std::cout << matrix[i] << " ";
                }
                std::cout << "..." << std::endl;
            } catch (const YAML::Exception& e) {
                std::cerr << "❌ 转换camera_matrix失败: " << e.what() << std::endl;
            }
        }
        
        // 检查复杂的嵌套结构
        if (yaml["detect"]) {
            std::cout << "✅ detect节点存在" << std::endl;
            if (yaml["detect"]["brightness"]) {
                std::cout << "  brightness节点存在" << std::endl;
            }
        }
        
    } catch (const YAML::Exception& e) {
        std::cerr << "❌ YAML解析错误: " << e.what() << std::endl;
        std::cerr << "错误位置: 行" << e.mark.line + 1 << ", 列" << e.mark.column + 1 << std::endl;
        
        // 打印错误行附近的内容
        std::istringstream content_stream(content);
        std::string line;
        int line_num = 0;
        while (std::getline(content_stream, line)) {
            line_num++;
            if (line_num >= e.mark.line - 2 && line_num <= e.mark.line + 2) {
                std::cout << (line_num == e.mark.line + 1 ? ">>> " : "    ") 
                         << line_num << ": " << line << std::endl;
            }
        }
    }
    
    std::cout << "\n--- 检查完成 ---\n" << std::endl;
}

int main() {
    checkYAML("configs/cs.yaml");
    return 0;
}