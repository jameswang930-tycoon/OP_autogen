# -*- coding: utf-8 -*-
"""第4部分 v6: 严格阶段0-7顺序, 策略只提六层+一例, Planner/Coder读入拆多块."""
import re
from pathlib import Path

base = Path(__file__).resolve().parent
p = base / 'architecture_mermaid_single.html'
t = p.read_text(encoding='utf-8')

new_block = '''<h2>4. 完整数据流图（阶段 0→7 顺序, 读入拆块）</h2>
<div class="mermaid">
%%{init: {'flowchart': {'curve': 'linear', 'nodeSpacing': 36, 'rankSpacing': 48}, 'themeVariables': {'fontSize': '15px'}}}%%
flowchart TD
    %% ═══ 阶段 0: 输入 ═══
    subgraph S0["阶段0 输入"]
        direction LR
        A1["源算子文件<br/>尺寸/精度/分块配置<br/>内核函数 + 测试驱动"]:::input
        A2["启动校验<br/>结构检查（只警告）"]:::input
        A3["基准数据<br/>PyTorch/工业级各模式"]:::input
    end

    %% ═══ 阶段 1: 采集 ═══
    subgraph S1a["阶段1 采集"]
        direction LR
        B1["预热裸跑<br/>JIT 编译+设备初始化"]:::diag
        B2["通用采集<br/>任务级 msprof 跑一遍"]:::diag
        B3["逐算子深度采集<br/>按算子名单逐个精采"]:::diag
    end
    B1out["运行日志<br/>崩溃原因可查"]:::out
    B2out["骨架数据<br/>耗时/核数/形状<br/>引擎占比/启动开销/L2"]:::out
    B3out["深度数据<br/>各级带宽/引擎利用率<br/>计算量/精度/冲突等待"]:::out

    subgraph S1b["阶段1 整合与分流"]
        direction LR
        B4["整合<br/>骨架+深度合并<br/>峰值算利用率"]:::diag
        B5["字段筛选<br/>按当前层筛"]:::diag
        B6["融合分析<br/>编译 → 依赖图"]:::diag
        B7["分块扫描<br/>枚举实测"]:::diag
    end
    B4out["诊断数据<br/>总耗时/算子数<br/>每算子瓶颈画像"]:::out
    B5out["层字段数据<br/>当前层专属字段"]:::out
    B6out["依赖图/融合候选"]:::out
    B7out["最优分块"]:::out

    %% ═══ 阶段 2: 设基准 ═══
    subgraph S2["阶段2 设基准（仅首轮）"]
        direction LR
        C1["复测源算子<br/>正确性 + 耗时"]:::diag
    end
    C1out["基线数据<br/>纯内核/端到端/事件<br/>+ 参考线"]:::out

    %% ═══ 阶段 3: 优化策略 ═══
    subgraph S3["阶段3 优化策略（共六层, 逐层执行）"]
        direction LR
        M0["六层优化策略<br/>算法结构/算子融合/分块配置<br/>访存/计算占用/架构专属"]:::input
        T1a["以算法结构层为例: 读<br/>计算量/利用率/精度占比<br/>瓶颈类型/算术强度"]:::doc
        T1b["判瓶颈<br/>算法选错/精度没吃满<br/>冗余访存"]:::decide
        T1c["改<br/>换算法/im2col/flash<br/>fp16 计算"]:::ok
        M0 --> T1a --> T1b --> T1c
    end

    %% ═══ 阶段 4a: Planner ═══
    subgraph S4a["阶段4a 策略制定（Planner）"]
        direction LR
        P1["读: 优化指导手册<br/>（skill）"]:::doc
        P2["读: 层字段数据<br/>（字段文件）"]:::doc
        P3["读: 当前算子源码"]:::doc
        P4["读: 历史轨迹"]:::doc
        P5["读: 优秀案例"]:::doc
        P6["读: 融合候选<br/>/分块结论"]:::doc
        P7["分析瓶颈出策略<br/>改动片段+理由<br/>预期收益+决策"]:::llm
    end
    P7out["策略方案<br/>修改指导方案"]:::out

    %% ═══ 阶段 4b: Coder ═══
    subgraph S4b["阶段4b 改码（Coder）"]
        direction LR
        C1c["读: 改码规范<br/>（skill）"]:::doc
        C2c["读: 策略方案"]:::doc
        C3c["读: 当前代码"]:::doc
        C4c["读: 失败案例库"]:::doc
        C5c["修改<br/>片段确定性替换"]:::llm
        C6c["修复<br/>报错+案例检索 → LLM<br/>成功回填"]:::llm
        C7c["校验<br/>语法/完整/非空"]:::diag
    end
    C8out["新代码 + 差异<br/>提交验证"]:::out

    %% ═══ 阶段 5: 验证 ═══
    subgraph S5["阶段5 验证"]
        direction LR
        F1["快速门<br/>正确性 + 事件快测"]:::decide
        F2["全量验证<br/>预热+正确性+测时<br/>事件多窗口中位"]:::diag
    end
    F2out["验证结果<br/>三口径耗时<br/>正确性结论"]:::out

    %% ═══ 阶段 6: 决策 + 迭代 ═══
    subgraph S6["阶段6 决策 + 迭代"]
        direction LR
        G1["比较<br/>端到端主加速比<br/>事件严格最优"]:::decide
        G2["采纳 / 回退"]:::ok
        G3["记录<br/>轨迹/案例/层推进<br/>基线校正"]:::doc
    end

    %% ═══ 阶段 7: 最终产物 ═══
    subgraph S7["阶段7 最终产物"]
        direction LR
        H1["最优算子<br/>+ 原始基线副本"]:::out
        H2["总结数据<br/>双口径加速比<br/>对比工业级"]:::out
        H3["轨迹图 + 策略记录"]:::out
    end

    %% ═══ 数据流主链 ═══
    A1 --> B1 --> B1out
    A1 --> B2 --> B2out
    A1 --> B3 --> B3out
    B2out --> B4
    B3out --> B4
    B4 --> B4out
    B4out --> B5 --> B5out
    B4out --> B6 --> B6out
    B4out --> B7 --> B7out
    A3 --> C1
    B5out --> C1
    C1 --> C1out
    B5out --> P2
    B5out --> T1a
    C1out --> P4
    B6out --> P6
    B7out --> P6
    P1 --> P7
    P2 --> P7
    P3 --> P7
    P4 --> P7
    P5 --> P7
    P6 --> P7
    P7 --> P7out
    P7out --> C2c
    C1c --> C5c
    C2c --> C5c
    C3c --> C5c
    C4c --> C5c
    C5c --> C7c --> C8out
    C6c -. 报错打回 .-> C5c
    C8out --> F1
    F1 -- 通过 --> F2 --> F2out
    F1 -- 不快于最优 --> G2
    F2out --> G1 --> G2 --> G3
    G3 --> H1
    G3 --> H2
    G3 --> H3
    G3 -. "迭代: 以当前最优进入下一轮" .-> B2

    classDef input fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px
    classDef diag fill:#f5f5f5,stroke:#9e9e9e
    classDef llm fill:#fff3e0,stroke:#fb8c00,stroke-width:2px
    classDef doc fill:#f3e5f5,stroke:#ab47bc
    classDef decide fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    classDef ok fill:#e8f5e9,stroke:#43a047
    classDef out fill:#e8f5e9,stroke:#43a047,stroke-width:2px
</div>'''

m = re.search(r'<h2>4\..*?</div>', t, re.S)
assert m, '未找到第4部分'
t = t[:m.start()] + new_block + t[m.end():]
p.write_text(t, encoding='utf-8')
print('第4部分 v6 已写入, 长度', len(new_block))