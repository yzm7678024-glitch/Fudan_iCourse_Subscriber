"""LLM-based course lecture summarization."""

import time

from openai import OpenAI

from src.runtime import config


SYSTEM_PROMPT = r"""你是一个专业的课程助教。你的任务是根据用户提供的课程录音文本和ppt文字ocr部分，生成用于学生自学和期末复习的详细笔记。
1. **直接输出**：不要包含任何"好的"、"没问题"、"以下是总结"等客套话，不要输出全局课程名称大标题（由系统自动生成），直接开始总结即可。
2. **文本清洗**：语言必须通顺、逻辑清晰，严格去除口语化表达、重复句和无意义的录音识别错误等。内容可能被识别成同音字，需要通过学术语境修复。
3. **格式严格**：
   - 必须使用 Markdown 格式排版。
   - 标题结构清晰与级别限制：只允许使用三级及后续级别的标题（即只能使用`###`、`####`或`#####`），禁止使用 `#` 和 `##`。用清晰的标题组织结构。
   - 不得使用超过两级的缩进和超过两级嵌套的bullet point。
   - 尽可能用完整的段落来组织老师的讲解，适量使用bullet point，不得过度使用bullet point
   - 合理使用加粗、列表、表格、分级标题、段首小标题等形式来组织信息，确保结构清晰。
4. **公式、数学推导与定量关系**：
   - 所有数学公式、物理公式、化学定量关系和科学变量必须使用规范的 LaTeX 语法书写（行内公式用 $...$，行间公式用 $$...$$）。由于图床限制，LaTeX 公式内部不要出现中文。
   - 对课程材料中出现的重要公式，应尽可能完整保留，不要仅用文字概括公式结论。
   - 对老师重点讲解、PPT 重点展示或具有明确推导过程的公式，应按照实际讲授逻辑尽可能完整整理，包括：出发公式或已知条件、使用的定义/定律/假设/近似、中间推导步骤以及最终公式或结论。
   - 推导过程中不要无故跳过关键步骤。对于重要的代入、微分、积分、展开、近似、变量替换、消元、极限处理、矩阵运算等，应尽量展示中间过程，并说明该变换的依据。
   - 对重要公式，应解释其中各符号所代表的物理量、化学量或数学对象；必要时注明单位、量纲、正负号约定、方向、边界条件以及适用范围。
   - 如果多个公式之间存在递进关系，应明确说明它们如何从前式得到后式，而不是孤立罗列公式。
   - 对容易混淆的公式，应说明其适用条件、与相近公式的区别以及常见错误。
   - 对课程中反复出现、作业或考试中可能直接使用的重要公式，可在相关章节末增加“关键公式”小节进行整理，但不能以公式汇总代替正文中的推导。
   - 对涉及图像、坐标系、矢量方向或正负号约定的公式，应结合文字说明其几何意义或物理意义。
   - 如果 OCR、字幕或录音转录中的公式明显残缺、符号冲突或无法可靠辨认，应结合上下文尽可能恢复；仍无法确认时，不得凭空编造，应明确标注“此处公式或符号需结合原 PPT 核对”。
   - 如果老师只口头使用“代入上式”“整理可得”“显然有”等省略表达，而根据已有上下文能够可靠还原中间步骤，可以适度补全推导；如果无法可靠确定，则不要猜测。
   - 对公式和推导密集的内容，应优先保证推导完整性和可理解性，允许明显增加篇幅。

5. **忠于原文、详略得当与篇幅要求**：
   - 总结必须有详有略，总体采用详细风格，以“学生能够尽量脱离录播，直接依靠笔记完成理解、自学和期末复习”为目标。
   - 篇幅应根据课程内容的信息密度自动调整，而不是机械追求固定的压缩比例。
   - 对寒暄、重复表述、口语化内容、无意义转录噪声以及音频与 PPT 之间重复出现的信息，应大胆压缩或合并。
   - 对核心概念、重要定义、教师重点解释、公式推导、复杂机理、算法过程、典型例题、实验原理、易错点和知识之间的逻辑联系，应充分保留和展开。
   - 对高知识密度内容，尤其是公式推导、复杂机理和重要例题，可以生成明显更长的笔记；不要为了缩短篇幅而删除关键推导、中间步骤、解释、例子或注意事项。
   - 反过来，如果某一部分课堂内容信息密度较低，也不要为了达到某个目标长度而人为扩写或重复表达。
   - 应包含原始材料中有助于理解的具体案例、文献、教师解释和推导细节，但不要机械逐句复述录音。
   - 必须忠于原始课程材料。禁止捏造录音、PPT 或上下文中没有依据的知识、结论、公式或教师要求；合理的结构整理、术语纠错以及能够由上下文可靠推出的中间推导不属于捏造。
6. 你需要格外注意课程中是否提及了作业、考试、签到、组队等关键事项，如果有的话，用三级标题【课程事项提醒】标注在开头。
7. **文风示例**：以下是一个关于"梯度下降"的片段，展示了笔记总结过程中【错误的】和【正确的】的两种总结风格，请严格模仿后者。

【❌ 错误的风格】
## 梯度下降

**定义：**
- 梯度下降是一种优化算法
- 用于最小化损失函数
- 广泛应用于机器学习

**核心步骤：**
- 计算梯度
- 更新参数
- 重复迭代

**学习率：**
- 学习率决定步长
- 太大会发散
- 太小会收敛慢
- 需要调参

**类型：**
- 批量梯度下降（BGD）
- 随机梯度下降（SGD）
- 小批量梯度下降（Mini-batch GD）

---

【✅ 正确的风格】

### 梯度下降

梯度下降是最小化损失函数 $L(\theta)$ 的核心优化算法。其基本思想是沿着损失函数对参数 $\theta$ 的梯度的反方向迭代更新，每一步的更新公式为 $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$，其中 $\eta$ 称为学习率，控制每次更新的步长大小。

**学习率的选取至关重要**：若 $\eta$ 过大，参数更新幅度过猛，损失函数可能在最优点附近震荡甚至发散；若 $\eta$ 过小，收敛速度极慢，训练成本大幅上升。实践中通常通过**学习率调度（learning rate schedule）**或**自适应方法（如 Adam）**来缓解这一问题。

根据每次更新时使用的样本量，梯度下降可分为三类：**批量梯度下降（BGD）** 每次使用全部训练数据，梯度估计准确但计算开销大；**随机梯度下降（SGD）** 每次仅用单个样本，更新频繁但噪声大；**小批量梯度下降（Mini-batch GD）** 则折中两者，是深度学习中最常用的形式。
---

8. **输入材料格式**：
   - 你收到的输入可能是两种格式之一：
     A) **带时间轴的分段格式**：用 `=== 时间段 mm:ss – mm:ss ===` 分隔每个 10 分钟段；段内有【音频转录】和【PPT 文字识别】两类内容。
     B) **平铺格式**：先一段【音频转录（无时间轴）】，后一段【PPT 文字识别（按出现顺序）】，每张 PPT 带 `[页 N @ mm:ss]` 标签。
   - 无论哪种格式，**PPT 文字是真实的板书 / 课件内容**，比音频识别噪声更可靠；遇到音频识别错的术语（同音字、专有名词），优先以 PPT 文字为准修正，但PPT的文字也可能有ocr的错误。
   - 但是，你的总结的组织的主线仍然应该以讲师的讲解为逻辑组织，同时把PPT的信息补充进去。
   - 同时，需要注意的是，由于该平台以屏幕截图的方式记录ppt，因此，尽管我们提供的ocr版本已经尽最大努力进行了清洗，你收到的ppt仍然可能有无关网页、导航栏、桌面或系统页面等无关噪音。这些噪音不应被用于课程内容的总结。
   - 输出仍按之前的格式要求，不要保留时间戳标签，只把这些信息当作上下文辅助理解，不用说哪些来自转写哪些来自ppt，自然地合并录音转写和ppt中的知识，生成高质量笔记。"""


class Summarizer:
    """Course lecture summarizer with multi-provider fallback.

    Iterates config.MODEL_PROVIDERS in declared order. Within each provider,
    tries each model in declared order. Returns the first successful result.

    DeepSeek official API supports selectable model and reasoning effort via
    runtime configuration.
    """

    def __init__(self):
        self.providers = config.resolve_model_providers()

        if not self.providers:
            raise ValueError(
                "No model provider available. "
                "Set at least one provider's API key "
                "(e.g. DEEPSEEK_API_KEY or DASHSCOPE_API_KEY)."
            )

        self._clients = {
            provider["name"]: OpenAI(
                api_key=provider["api_key"],
                base_url=provider["base_url"],
            )
            for provider in self.providers
        }

    def _call_llm(
        self,
        client: OpenAI,
        model: str,
        title: str,
        content: str,
        reasoning_effort: str | None = None,
    ) -> str:
        """Call one LLM model and return a non-empty summary."""

        t0 = time.time()

        request_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"以下是课程《{title}》的录音转录文本和 PPT OCR 内容。"
                        f"请根据内容的信息密度生成详细、完整、适合自学和期末复习的课程笔记。"
                        f"本次可将约 {len(content) // 5} 字符作为一般性的输出长度参考，"
                        f"但这只是软性参考而不是必须达到的目标。"
                        f"如果材料中存在大量重复、口语化内容或音频与 PPT 的重复信息，可以适当短于该长度；"
                        f"如果包含较多公式推导、复杂机理、算法过程、典型例题或教师重点解释，"
                        f"则可以明显超过该长度。"
                        f"不要为了控制篇幅而省略重要知识，也不要为了达到目标长度而人为扩写。"
                        f"请开始总结：\n\n{content}"
                    ),
                },
            ],
            "timeout": 180,
        }

        # DeepSeek 官方 API 专属配置。
        #
        # deepseek-v4-flash / deepseek-v4-pro 默认启用思考模式，
        # 这里显式开启，并根据 config.py 中的设置控制推理强度。
        #
        # ModelScope 等其他 OpenAI-compatible provider 不会进入这里，
        # 因此不会收到 DeepSeek 官方接口专属参数。
        if model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            request_kwargs["reasoning_effort"] = (
                reasoning_effort
                or config.DEEPSEEK_REASONING_EFFORT
            )
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                }
            }

        # 真正调用模型 API。
        response = client.chat.completions.create(**request_kwargs)

        # API 请求成功并不代表一定返回了有效内容。
        if not response.choices:
            raise ValueError(
                "API returned empty choices — "
                "likely content filter, quota, or provider error"
            )

        result = response.choices[0].message.content

        # 修复旧版本 Bug：
        # 如果模型返回 None、空字符串或纯空白，
        # 必须视为调用失败，不能将该课次标记为处理完成。
        #
        # 异常会被 summarize() 捕获，并继续尝试备用 provider/model；
        # 如果全部失败，则 LectureRunner 会保留该课次供以后重试。
        if not result or not result.strip():
            raise ValueError("API returned an empty summary")

        result = result.strip()
        elapsed = time.time() - t0

        # 输出 token 使用情况，便于判断成本与调用规模。
        usage = getattr(response, "usage", None)

        if usage is not None:
            prompt_tokens = getattr(
                usage,
                "prompt_tokens",
                "?",
            )
            completion_tokens = getattr(
                usage,
                "completion_tokens",
                "?",
            )

            print(
                f"[Summarizer] Done ({model}): "
                f"{len(content)} chars input → "
                f"{len(result)} chars output "
                f"in {elapsed:.0f}s "
                f"(tokens: prompt={prompt_tokens}, "
                f"completion={completion_tokens})"
            )
        else:
            print(
                f"[Summarizer] Done ({model}): "
                f"{len(content)} chars input → "
                f"{len(result)} chars output "
                f"in {elapsed:.0f}s"
            )

        return result
       
    def summarize(
        self,
        title: str,
        content: str,
        deepseek_model: str | None = None,
        deepseek_reasoning_effort: str | None = None,
    ) -> tuple[str, str]:
           """Summarize lecture, trying providers in configured order.
   
           deepseek_model and deepseek_reasoning_effort apply only to the
           official DeepSeek provider. Other fallback providers keep their
           existing configured model order.
   
           Returns:
               (summary, model_used)
   
               model_used format:
               "{provider}/{model}"
   
           Raises:
               RuntimeError:
                   If every configured provider/model fails.
           """
   
           if not content or not content.strip():
               return ("（内容为空）", "")
   
           errors = []
   
           for provider in self.providers:
               client = self._clients[provider["name"]]
   
               # For the official DeepSeek provider, allow this individual
               # lecture/course to override the global default model.
               if provider["name"] == "deepseek" and deepseek_model:
                   models = [deepseek_model]
               else:
                   models = provider["models"]
   
               for model in models:
                   model_id = f"{provider['name']}/{model}"
   
                   try:
                       result = self._call_llm(
                           client,
                           model,
                           title,
                           content,
                           reasoning_effort=deepseek_reasoning_effort,
                       )
   
                       return (result, model_id)
   
                   except Exception as exc:
                       print(
                           f"[Summarizer] {model_id} failed: "
                           f"{type(exc).__name__}: {exc}"
                       )
   
                       errors.append(
                           f"{model_id}: "
                           f"{type(exc).__name__}: {exc}"
                       )
   
           raise RuntimeError(
               "All LLM models failed:\n"
               + "\n".join(errors)
           )
