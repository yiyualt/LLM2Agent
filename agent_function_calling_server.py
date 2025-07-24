# agent_function_calling_server.py
"""
最小可运行的 function calling agent server/client 架构示例。
server 端用 Flask，client 端用 requests。
"""
# server.py
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 定义工具
@tool("delete_file", return_direct=True)
def delete_file(filename: str) -> str:
    """删除指定文件"""
    import os
    if not os.path.exists(filename):
        return f"文件 {filename} 不存在。"
    os.remove(filename)
    return f"已删除文件 {filename}。"

@tool("create_file", return_direct=True)
def create_file(filename: str, content: str = "") -> str:
    """创建新文件并写入内容"""
    import os
    if os.path.exists(filename):
        return f"文件 {filename} 已存在。"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"已创建文件 {filename}。"

tools = [delete_file, create_file]

# 初始化 LLM 和 agent
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=api_key
)
agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

app = Flask(__name__)

@app.route('/agent', methods=['POST'])
def agent_api():
    try:
        user_input = request.json['input']
        result = agent.run(user_input)
        return jsonify({"result": result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)

# client.py 示例：
'''
import requests

resp = requests.post('http://localhost:5000/agent', json={'input': '创建文件 demo.txt 内容为 hello server'})
print(resp.json())

resp = requests.post('http://localhost:5000/agent', json={'input': '删除文件 demo.txt'})
print(resp.json())
''' 