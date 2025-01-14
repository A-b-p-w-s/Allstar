"""
没用的
"""
import pyautogui
import time
import random

try:
    while True:
        # 模拟按下键
        pyautogui.press('down')
        # 等待1秒
        random_time = random.uniform(1, 5)
        time.sleep(random_time)
except KeyboardInterrupt:
    print("程序被用户中断")