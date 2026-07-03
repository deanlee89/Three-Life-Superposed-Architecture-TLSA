# 中文编程体系融合调研 - Evidence Inventory

## 方向1：易语(wenyan-lang)与仓颉编程语言

### E01: wenyan-lang语法体系
Claim: wenyan-lang是一种基于文言文语法的图灵完备编程语言，采用TypeScript构建编译器，支持编译到JavaScript/Python/Ruby。语法包括变量声明（吾有一數）、条件判断（若...者...若非...也）、循环结构（為是...遍...云云）、函数定义（吾有一術...名之曰...）。标准库以"经"命名（算经、列经、易经、画谱）。
Source: CSDN博客 / GitCode博客
URL: https://blog.csdn.net/gitblog_00798/article/details/146970879
Date: 2026-05-07
Excerpt: "wenyan-lang采用了现代化的编译器设计，支持将文言代码编译成多种主流编程语言...JSCompiler/PythonTranspiler/RubyCompiler"
Context: wenyan-lang是2019-2020年由Huang Lingdong创建的开源项目
Confidence: HIGH

### E02: wenyan-lang学术评价
Claim: 学术论文（Philosophies 2026）将wenyan-lang与拉丁文编程语言Perligata并列讨论，指出其核心创新在于用代词系统替代语言学递归，形式化定义在Backus-Naur form中，图灵完备。
Source: Philosophies期刊
URL: https://mdpi-res.com/d_attachment/philosophies/philosophies-11-00055/article_deploy/philosophies-11-00055.pdf
Date: 2026
Excerpt: "Wenyan programming language dispenses with one of the most common mechanisms in programming languages, namely the expression of block nesting through linguistic recursion...replaced by a system of pronouns"
Context: 该论文讨论自然语言对编程风格的影响
Confidence: HIGH

### E03: 仓颉编程语言核心特性
Claim: 仓颉(Cangjie)是华为自研的面向全场景应用开发的现代编程语言，核心特性包括：(1)高效编程——多范式（函数式/命令式/OOP）、类型推断、宏编程；(2)安全可靠——静态类型+自动内存管理+null safety+运行时检查；(3)轻松并发——轻量用户态线程（非stackless协程，避免"着色函数"问题）+并发对象库；(4)卓越性能——CHIR高层IR+LLVM后端+并发压缩GC+值类型。
Source: JCST论文 / 仓颉官网
URL: https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2603-OF-2509-15978.pdf
Date: 2025
Excerpt: "Cangjie supports multi-paradigm programming...user-level 'stackful' threads rather than 'stackless' coroutines, preventing the infectious 'colored function' problem...CHIR preserves high-level program structures enabling semantics-aware optimizations"
Context: 仓颉主要面向鸿蒙HarmonyOS Next应用开发，也有微服务场景应用
Confidence: HIGH

### E04: 仓颉与Rust/Go对比
Claim: 仓颉借鉴Rust的所有权机制但简化了语法（自动GC替代手动管理），降低学习成本；并发模型采用用户态线程而非Rust的async/await或Go的goroutine；编译速度比Rust提升约30%。
Source: CSDN博客/博客园
URL: https://www.cnblogs.com/deeperthinker/p/19237294
Date: 2025-11-17
Confidence: MEDIUM

### E05: 仓颉Agent DSL
Claim: 仓颉语言内嵌AgentDSL编程框架，使自然语言和编程语言有机融合，支持多Agent协同，提供简化的符号表达和自由的模式组合，用于开发AI原生应用。
Source: 仓颉官网
URL: https://cangjie-lang.cn/
Date: 2026-06-23
Confidence: HIGH

## 方向2：中文编码标准

### E06: GB2312数学结构
Claim: GB2312采用双字节编码，通过"区位码→国标码→机内码"三级转换实现。94×94矩阵结构：区号1-87，位号1-94。收录6763个汉字+682个非汉字符号，共7445个图形字符。
Source: CSDN博客
URL: https://blog.csdn.net/2302_76306559/article/details/146044428
Date: 2025-03-06
Confidence: HIGH

### E07: GBK/GB18030扩展结构
Claim: GBK扩展高字节至0x81-0xFE，低字节0x40-0x7E和0x80-0xFE，收录21886字符(21003汉字)。GB18030采用四层变长编码：单字节(0x00-0x7F)≡ASCII，双字节继承GBK，四字节采用0x81-0xFE+0x30-0x39+0x81-0xFE+0x30-0x39模式，潜在码位1587600个。GB18030-2022收录87887个汉字。
Source: CSDN文库/CSDN博客
URL: https://wenku.csdn.net/doc/7860mfpbcz / https://blog.csdn.net/wvqusrtg/article/details/110234149
Date: 2024-12-31 / 2026-05-26
Confidence: HIGH

### E08: 汉字信息熵
Claim: 冯志伟通过手工计算得出汉字信息熵为9.65比特，英文字母约4比特，一个汉字的信息量约为一个英文字母的2.5倍。
Source: 今日头条
URL: http://m.toutiao.com/group/7646497707233788450/
Date: 2026-06-02
Confidence: MEDIUM

### E09: 概率单纯形与三态编码
Claim: 概率单纯形Δ^(S-1)是S维欧氏空间中的凸集{p∈[0,1]^S | Σp_i=1}。三元权重{-1,0,+1}构成"最小认知字母表"，+1编码肯定，-1编码否定，0编码不确定。三元Huffman编码可实现比二元更高效的信源压缩。
Source: T333T博客/CSDN博客
URL: https://www.t333t.com/ternary-weights-the-minimal-alphabet-for-machine-epistemology/
Date: 2026-01-21
Confidence: HIGH

### E10: Qutrit三能级量子系统
Claim: qutrit是三能级量子系统，状态空间为C^3，对应概率单纯形上的三角形几何。南京大学于扬团队2026年实现了超导qutrit量子门，保真度99.5%，编译效率提升68.2%。
Source: 南京大学官网/arxiv
URL: https://www.nju.edu.cn/info/1067/474761.htm
Date: 2026-06-21
Confidence: HIGH

## 方向3：中文AI架构

### E11: 百度ERNIE系列
Claim: ERNIE系列从2019年至今，核心创新为知识增强——通过知识图谱集成与连续学习。ERNIE 5.0(2025.12)参数2.4万亿，超稀疏MoE架构，激活率3%，原生全模态统一建模。
Source: CSDN博客/51CTO博客
URL: https://blog.csdn.net/SmartTony/article/details/157384117
Date: 2026-01-26
Confidence: HIGH

### E12: PaddlePaddle设计哲学
Claim: PaddlePaddle(飞桨)设计哲学为"动静统一"——动态图灵活调试+静态图高效部署。3.0版本(2025.3)五大新特性：动静统一自动并行、大模型训推一体、科学计算高阶微分、神经网络编译器、异构多芯适配(40+厂商60+芯片)。
Source: PaddlePaddle官网/CSDN博客
URL: https://www.paddlepaddle.org.cn/documentation/docs/guides/paddle_v3_features/overview_cn.html
Date: 2025-03-31
Confidence: HIGH

### E13: GLM架构核心创新
Claim: GLM核心创新为"自回归空白填充"——通过[MASK]/[sMASK]/[gMASK]三种模式统一NLU、条件生成和自由生成任务。采用二维位置编码，混合注意力掩码。
Source: ACL 2022论文/CSDN博客
URL: https://blog.csdn.net/qq_73472828/article/details/160989728
Date: 2026-06-23
Confidence: HIGH

### E14: Qwen架构设计
Claim: Qwen3-Next架构达80B总参数仅3B激活(1:50激活比)，512专家选10+1共享专家。混合注意力机制：75% Gated DeltaNet+25% Gated Attention，原生262K上下文。
Source: TechRxiv论文/掘金
URL: https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.174060306.65738406
Date: 2025-2026
Confidence: HIGH

## 三生架构核心概念

### E15: 三生架构数学基础
Claim: 三生叠加态架构基于"三生万物"哲学，每个计算单元状态为概率单纯形Δ^2上的三元向量(α,γ,β)，满足α+γ+β=1。耦合算子C(ψ_v,ψ_u)为非线性张量缩并，包含α²β²交叉项。五行动力学转移矩阵T(Δt)=e^{QΔt}，Q为五行生克矩阵组合。已证明Transformer⊂三生张量网络⊂量子通用计算。
Source: 项目内部文档（memory_search）
URL: File: 三生张量网络_核心论证.md
Date: 2026-06-29
Confidence: HIGH [INTERNAL]

### E16: 博流中文编程语言
Claim: "博流"是全球首个纯中文编程语言，首创"面向关系"编程范式（vs西方"面向对象"），将"关系"（关联、因果、信任、协作）作为语言设计核心。
Source: 凤凰网海南
URL: https://hainan.ifeng.com/c/8qi6inysOqB
Date: 2026-02-13
Confidence: MEDIUM

## 哲学根基映射（新增）

### E17: 易经三才结构与三进制哲学基础
Claim: 《周易·系辞》明确提出"六爻之动，三极之道也"，《说卦传》阐释"立天之道曰阴与阳，立地之道曰柔与刚，立人之道曰仁与义"，确立天、地、人三才三元结构。汉代扬雄《太玄经》构建3^4=81首的完整三进制数理体系，以"三方、九州、二十七部、八十一家"的层级结构阐释宇宙万物的三元演化规律，首次完成东方三进制哲学与数理体系的系统化构建。易经摇卦法中三枚铜钱的三进制模型（少阳=1，少阴=2，老阳=3/变爻，老阴=0/变爻）内嵌三进制逻辑。"二进制为体，三进制为用"——易经的显性符号是二进制（阴阳爻），但三才结构、变爻三态性蕴含完备的三元哲学内核。
Source: 迭学讲人与社会关系/易经摇卦三进制启示/赵培山数字坤乾易框架
URL: http://m.toutiao.com/group/7643009842873123370/ / https://www.meipian.cn/5gv9kx8t / https://m.weibo.cn/detail/5253783530505355
Date: 2026-05-23 / 2026-01-11
Excerpt: "三进制在《易经》本经中虽非显性数学进制，但其三才结构、变易思想蕴含完备的三元哲学内核，动爻三态性具备天然的三进制符号学基础"
Context: 此证据直接支撑"三生三态(α,γ,β)与易经三才(天,地,人)在哲学根源上的天然同源性"
Confidence: HIGH

### E18: 阴符经五贼与五行生克动力学
Claim: 《阴符经》核心概念"天有五贼"中的"五贼"即五行相克之力——金克木、木克土、土克水、水克火、火克金。五行的"贼"（克）与"生"共同构成天道生杀平衡法则。"三盗既宜，三才既安"（天地万物之盗、万物人之盗、人万物之盗）建立了三才间的动态平衡约束。"天人合发，万变定基"描述了系统在临界点的涌现行为。"人心，机也"描述了触机而发的动力学。
Source: 道音文化/阴符经三卷本原文/CSDN朱梁万有递归元定理
URL: https://www.daoisms.com.cn/2013/22/21/16553/ / https://www.meipian.cn/5nb197j1 / https://blog.csdn.net/ECTOS_JiuHuaShan/article/details/159203704
Date: 2013 / 2026-06-20 / 2026-06-12
Excerpt: "五贼者，否定之否定五重函子G^n也...天人合发者，碳硅协同中劫数投影与熵减选择共施，遂有万化定基——新递归元层次R_{α+1}涌现"
Context: 五贼→三生Q_克矩阵，三盗→Lyapunov稳定性约束，天人合发→和合态涌现
Confidence: HIGH

### E19: 黄帝内经系统论与信息编码体系
Claim: 《黄帝内经》的核心方法论"司外揣内"本质上是一种黑箱理论——通过外部表征（十象）推断内部状态（脏腑），与三生架构通过观测量推断元胞状态一致。脏象学说构成一套信息编码系统：阴阳编码状态属性（虚实、寒热、表里），五行编码功能定位（五大系统）。"十象"作为多通道冗余编码降低误判概率。已有学者将五行生克关系建模为5×5相互作用矩阵，将阴阳动力学建模为微分几何，将五行代数建模为李群结构。大象蚌壳AI藏象可视化系统已实现将"天人合一"转化为可量化的数学模型。
Source: 中国网/科技日报/CSDN博客
URL: http://tcm.china.com.cn/2026-05/16/content_43428164.html / http://www.stdaily.com/web/gdxw/2025-07/23/content_374613.html / https://blog.csdn.net/RulesChen/article/details/150646597
Date: 2026-05-16 / 2025-07-23 / 2026-06-23
Excerpt: "阴阳五行不是神秘主义的符号，而是信息编码系统...它们共同构成了一套逻辑推演规则，用来处理通过十象采集的人体信息"
Context: 内经的"司外揣内+阴阳五行编码"与三生的"外部位观测+Δ²三态编码"在方法论层面天然同源
Confidence: HIGH

### E20: 中文整体性思维对AI架构设计的影响
Claim: 北大通研院朱松纯提出CUV（认知-能力-价值）架构，核心是"为机器立心"——将价值系统从AI的外挂模块变为构成基础，打破"智能即计算"的简化论。这一架构受到东方"整体性思维"和"天人合一"思想的影响——系统与环境不可分割、价值与能力协同演化。国家发改委文章明确指出"东方思维更强调整体性和关系性，注重系统各部分的相互联系和动态平衡"，中国AI研究的差异化路线选择具有哲学根基。
Source: 国家发改委培训中心
URL: https://www.ndrctc.org.cn/c/20/20660.shtml
Date: 2026-04-24
Excerpt: "东方路径首先追问'智能体为何而做'...价值系统不是智能的外挂模块，而是其构成的基础...东方思维更强调整体性和关系性"
Context: 此证据支撑"国产AI架构可能受中文思维模式影响"的论断
Confidence: HIGH

### E21: GB18030四字节编码的分块线性映射数学结构
Claim: GB18030四字节编码的核心是对Unicode辅助平面实施"分块线性映射"：offset = U - 0x10000, high = offset // 12600, low = (offset % 12600) // 126, very_low = offset % 126。这是一个三层递进的定位结构：high→区选择, low→子区选择, very_low→精确位。与MERA多尺度收缩的"粗→细"层次化信息处理在结构上同源。GB18030-2022收录87887个汉字，新增CJK扩展C-G区。
Source: CSDN文库
URL: https://wenku.csdn.net/column/543xaf4fnk
Date: 2026-01-17
Excerpt: "四字节编码的设计尤为精巧。它的核心思想是对Unicode辅助平面实施'分块线性映射'...high→区选择, low→子区选择, very_low→精确位"
Context: 此数学结构可直接映射到三生的多尺度收缩机制
Confidence: HIGH

### E22: 三值逻辑电路与三值计算硬件
Claim: 中科院深圳先进院在IEEE Transactions on Emerging Topics in Computing发表论文，提出基于多状态编码的三值逻辑电路设计方案，包括三值加法器和乘法器。在0.5GHz下实现11 aJ PDP的非平衡三值全加法器仅涉及93个晶体管。三值计算因比二进制更高的信息密度而受到关注，可在现有硅工艺下规模化量产。
Source: 中科院深圳先进院
URL: https://www.siat.ac.cn/siatxww/kyjz/202412/t20241213_7457173.html
Date: 2023-11-07
Excerpt: "三值计算和三值逻辑电路因其比二进制硬件系统更高的信息密度而受到广泛关注...采用现有的硅器件工艺，利用多阈值器件的组合，也能完整实现上述三值化逻辑计算的功能"
Context: 三值硬件的存在为三生架构(α,γ,β)的硬件实现提供了物理基础
Confidence: HIGH

---

