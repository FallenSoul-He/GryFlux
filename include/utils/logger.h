/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <vector>
#include <chrono>
#include <iomanip>
#include <atomic>
#include "utils/noncopyable.h"

namespace GryFlux
{

    // 日志级别
    enum class LogLevel
    {
        TRACE,
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        FATAL,
        OFF
    };

    // 日志输出目标
    enum class LogOutputType
    {
        CONSOLE,
        FILE,
        BOTH
    };

    class Logger
        : private NonCopyableNonMovable
    {
    public:
        // 获取Logger单例
        static Logger &getInstance();

        // 配置日志级别
        void setLevel(LogLevel level);

        // 配置日志输出目标
        void setOutputType(LogOutputType type);

        // 配置LOG所属应用名称
        void setAppName(const std::string &appName);

        // 设置日志文件
        bool setLogFileRoot(const std::string &fileRoot);

        // 显示时间戳
        void showTimestamp(bool show);

        // 显示日志级别
        void showLogLevel(bool show);

        // 统一的字符串记录方法
        void logString(LogLevel level, const std::string &message)
        {
            if (level < currentLevel_)
                return;
            writeLog(level, message);
        }

        // 日志记录方法
        template <typename... Args>
        void log(LogLevel level, const char *format, Args &&...args)
        {
            if (level < currentLevel_)
                return;

            std::string message;
            try
            {
                message = formatString(format, std::forward<Args>(args)...);
            }
            catch (const std::exception &e)
            {
                std::stringstream ss;
                ss << "格式化日志失败: " << e.what() << " (原始消息: " << format << ")";
                message = ss.str();
            }
            writeLog(level, message);
        }

        // 无参数版本的log方法
        void log(LogLevel level, const char *message)
        {
            if (level < currentLevel_)
                return;
            writeLog(level, std::string(message));
        }

        // 辅助方法：各种日志级别
        template <typename... Args>
        void trace(const char *format, Args &&...args)
        {
            log(LogLevel::TRACE, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void debug(const char *format, Args &&...args)
        {
            log(LogLevel::DEBUG, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void info(const char *format, Args &&...args)
        {
            log(LogLevel::INFO, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void warning(const char *format, Args &&...args)
        {
            log(LogLevel::WARNING, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void error(const char *format, Args &&...args)
        {
            log(LogLevel::ERROR, format, std::forward<Args>(args)...);
        }

        template <typename... Args>
        void fatal(const char *format, Args &&...args)
        {
            log(LogLevel::FATAL, format, std::forward<Args>(args)...);
        }

        // 无参数版本
        void trace(const char *message) { log(LogLevel::TRACE, message); }
        void debug(const char *message) { log(LogLevel::DEBUG, message); }
        void info(const char *message) { log(LogLevel::INFO, message); }
        void warning(const char *message) { log(LogLevel::WARNING, message); }
        void error(const char *message) { log(LogLevel::ERROR, message); }
        void fatal(const char *message) { log(LogLevel::FATAL, message); }

    private:
        Logger();
        ~Logger();

        // 获取当前时间戳
        std::string getCurrentTimestamp();

        // 获取日志级别的字符串表示
        std::string getLevelString(LogLevel level);

        // 生成日志文件名
        std::string generateLogFileName(const std::string &prefix);

        // 写入日志
        void writeLog(LogLevel level, const std::string &message);

        LogLevel currentLevel_;
        LogOutputType outputTarget_;
        std::ofstream logFile_;
        std::mutex logMutex_;
        bool showTimestamp_;
        bool showLogLevel_;
        std::string app_name_;

        // 处理std::atomic类型的辅助函数
        template <typename T>
        T get_value(const std::atomic<T> &a)
        {
            return a.load();
        }

        // 通用类型处理
        template <typename T>
        T get_value(const T &t)
        {
            return t;
        }

        // 特殊处理字符数组
        template <size_t N>
        const char *get_value(const char (&arr)[N])
        {
            return arr;
        }

        // 处理非const字符数组
        template <size_t N>
        const char *get_value(char (&arr)[N])
        {
            return arr;
        }

        // 使用变参模板递归解包并处理std::atomic类型
        template <typename... Args>
        std::string formatString(const char *format, Args &&...args)
        {
            return formatStringImpl(format, get_value(std::forward<Args>(args))...);
        }

        // 实际的格式化实现
        template <typename... Args>
        std::string formatStringImpl(const char *format, Args... args)
        {
            // 初始尝试，为了确定所需的缓冲区大小
            int size_s = std::snprintf(nullptr, 0, format, args...) + 1; // 额外空间用于空终止符
            if (size_s <= 0)
            {
                return "格式化错误";
            }

            // 分配缓冲区
            auto size = static_cast<size_t>(size_s);
            std::unique_ptr<char[]> buf(new char[size]);

            // 实际进行格式化
            std::snprintf(buf.get(), size, format, args...);

            // 返回格式化后的字符串
            return std::string(buf.get(), buf.get() + size - 1); // 不包含空终止符
        }

        // 无参数版本的formatString
        std::string formatString(const char *format)
        {
            return std::string(format);
        }
    };
// 全局日志对象
#define LOG GryFlux::Logger::getInstance()
} // namespace GryFlux
