"""T13-2 gate: launch() 目录式适配（三个必须交代的细节）。

- new_run_id() 产出唯一 id（多轮隔离的基石）
- 模块 docstring 交代三点：① 目录可配置 ② run-id 隔离 ③ 等待与超时
- launch() 槽位注释指向这三点
"""
from control import launch_template as lt


def test_new_run_id_unique():
    a = lt.new_run_id()
    b = lt.new_run_id()
    c = lt.new_run_id()
    assert a != b != c != a
    assert a.startswith("run_")


def test_module_docstring_documents_configurable_dirs():
    doc = lt.__doc__
    assert ("配置" in doc or "环境变量" in doc) and "硬编码" in doc


def test_module_docstring_documents_run_id_isolation():
    doc = lt.__doc__.lower()
    assert ("run id" in doc or "run_id" in doc or "隔离" in doc)


def test_module_docstring_documents_wait_and_timeout():
    doc = lt.__doc__
    assert ("超时" in doc or "timeout" in doc.lower()) and ("轮询" in doc or "等待" in doc)


def test_launch_slot_comment_points_to_constraints():
    # the launch() docstring must reference the directory-mode constraints
    doc = lt.launch.__doc__ or ""
    assert "run id" in doc.lower() or "run_id" in doc.lower() or "目录" in doc
