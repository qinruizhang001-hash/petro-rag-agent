"""
tests —— 测试包。

有这个文件，``python -m unittest discover -s tests`` 才能工作：``discover``
要求起始目录是可导入的包，而 README 里就是这么写的命令。CI 不用 discover
（它逐个点名模块，见 .github/workflows/ci.yml），所以这个文件只服务于本地。
"""
