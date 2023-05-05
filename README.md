### QA-chatbot-workshop

- 代码介绍

```python
.
├── code
│   ├── main.py                          # lambda 部署主文件
│   ├── aos_write_job                    # aos 倒排和knn索引构建脚本 (glue 部署)
│   ├── kendra_write_job.py              # kendra数据导入脚本 (glue 部署)
│   ├── chatbot_logs_func.py             # 对Cloudwatch输出的日志解析，通过KDF同步到OpenSearch (lambda 脚本)
│   ├── offline_trigger_lambda.py        # 调度 glue 的 lambda 脚本
│   ├── QA_auto_generator.py             # 基于文档自动生成FAQ知识库 (离线前置处理)
│   └── kendra_write_job.py              # kendra数据导入脚本 (glue 部署)
├── docs
│   ├── Cleanroom_FAQ.txt                # 知识库文件
│   └── EMR_Best_Practice_FAQ.txt        # EMR Best Practice 知识
├── AWS_DOC_POC.ipynb                    # AWS文档-本地效果测试 notebook
├── document_segment.ipynb               # 文档切分调优 开发notebook
├── lanchain_demo.ipynb                  # lanchain 开发notebook
├── langchain+basic.ipynb                # lanchain 开发notebook
├── chatglm_deploy.ipynb                 # chatglm LLM Model 部署notebook
├── bloomz_LLM_deploy.ipynb              # bloomz LLM Model 部署notebook
├── paraphrase-multilingual-deploy.ipynb # Sentence2Embedding Model 部署notebook
├── Local_SentenceEmb.ipynb              # Studio 模型部署调试 notebook
└── SentenceEmbedding_deploy.ipynb       # GPT-6J Embedding Modeljumpstart部署 notebook
```

- 流程介绍

  - 离线流程
    - 前端界面上传文档zip
    - 清空AOS
    - 清空Kendra
  - 在线流程
    - 前端[网页](http://chatbot-alb-1653663846.us-east-1.elb.amazonaws.com:9988/)直接聊天
    - 前端[网页](http://chatbot-alb-1653663846.us-east-1.elb.amazonaws.com:9988/)切换模型

- 系统架构

  png