import os
import logging

def setup_logger(logger_name, save_path):
    # 创建一个日志器
    logger = logging.getLogger(logger_name)  # 名字为'gift'的日志器

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 设置日志器的日志等级
    logger.setLevel(logging.DEBUG)

    # 创建一个控制台输出 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个文件输出 handler
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(logging.DEBUG)

    # 创建一个日志格式器
    formatter = logging.Formatter('[%(asctime)s %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加 handler 到日志器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

