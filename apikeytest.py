# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:54:45 2026
AIzaSyAoqOb5VV9xRaXaLR2GoNDV-xtoiNrCXNE
@author: yiting
"""

from google import genai

# 1. 在這裡貼上你的 API Key
YOUR_API_KEY = "這裡換成你的API_KEY"

def check_gemini_key():
    try:
        # 初始化新版 Client
        client = genai.Client(api_key="AIzaSyAoqOb5VV9xRaXaLR2GoNDV-xtoiNrCXNE")
        
        print("正在測試新版 API Key 連線...")
        
        # 新版 SDK 的呼叫方式更簡潔
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents="你好，請說『測試成功』"
        )
        
        print("-" * 30)
        print("✅ 檢測成功！你的 API Key 是完全正常的。")
        print(f"Gemini 回應：{response.text}")
        print("-" * 30)
        
    except Exception as e:
        print("-" * 30)
        print("❌ 檢測失敗！")
        print(f"詳細錯誤訊息：{e}")
        print("-" * 30)

if __name__ == "__main__":
    check_gemini_key()