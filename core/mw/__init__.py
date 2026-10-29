"""自建中间件集合。

命名用 `mw` 而不是 `middleware`，是为了在 import 时和官方包
`langchain.agents.middleware` 一眼区分开。

【中间件必须是"无状态"的】
AGENT 在 core/agent.py 模块级构建一次，这里的所有实例被所有请求共享。
任何随请求变化的计数/标识都必须走 core/run_state 的 ContextVar，
不能写在 self 上 —— 否则并发时会串号（A 的步数让 B 熔断）。
实例属性只允许放不可变配置。
"""
