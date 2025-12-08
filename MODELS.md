# AI Model Integration Guide

**Document Type**: AI Model Specifications & Integration Manual  
**Target Audience**: AI Agent Developers, Backend Engineers  
**Last Updated**: December 2025

---

## Table of Contents

### OpenRouter Provider (Free Tier)
1. [Alibaba Tongyi DeepResearch 30B](#alibaba-tongyi-deepresearch-30b)
2. [AllenAI OLMo 3 32B Think](#allenai-olmo-3-32b-think)
3. [OpenAI GPT-OSS 120B](#openai-gpt-oss-120b)
4. [OpenAI GPT-OSS 20B](#openai-gpt-oss-20b)
5. [TNG DeepSeek R1T2 Chimera](#tng-deepseek-r1t2-chimera)
6. [Z.AI GLM 4.5 Air](#zai-glm-45-air)
7. [NVIDIA Nemotron Nano 9B v2](#nvidia-nemotron-nano-9b-v2)
8. [Meituan LongCat Flash Chat](#meituan-longcat-flash-chat)
9. [Google Gemma 3 27B IT](#google-gemma-3-27b-it)

### NVIDIA NIM Provider (Paid Tier)
10. [Meta Llama 3.1 405B Instruct](#meta-llama-31-405b-instruct)
11. [Meta Llama 3.1 70B Instruct](#meta-llama-31-70b-instruct)
12. [Meta Llama 3.1 8B Instruct](#meta-llama-31-8b-instruct)
13. [MistralAI Mistral Nemotron](#mistralai-mistral-nemotron)
14. [Qwen Qwen2.5 7B Instruct](#qwen-qwen25-7b-instruct)

---

# OpenRouter Provider

## Alibaba Tongyi DeepResearch 30B

**Model ID**: `alibaba/tongyi-deepresearch-30b-a3b:free`

**Specifications**:
- Context: 32,768 tokens
- Modality: Text → Text
- Cost: Free tier
- Use Case: Complex reasoning, research, multi-step analysis

**Client Initialization**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY"
)
```

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="alibaba/tongyi-deepresearch-30b-a3b:free",
    messages=[{"role": "user", "content": "How many r's are in 'strawberry'?"}],
    reasoning={"enabled": True},  # Enable reasoning mode
    max_tokens=2048,
    temperature=0.3
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Always enable `reasoning: {"enabled": True}` for complex tasks
- Preserve `reasoning_details` in multi-turn conversations
- Expect 5-15s latency for reasoning tasks
- Best for step-by-step analysis

---

## AllenAI OLMo 3 32B Think

**Model ID**: `allenai/olmo-3-32b-think:free`

**Specifications**:
- Context: 32,768 tokens
- Modality: Text → Text
- Cost: Free tier
- Use Case: Academic research, educational content, analytical reasoning

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="allenai/olmo-3-32b-think:free",
    messages=[{"role": "user", "content": "Solve this mathematical problem."}],
    reasoning={"enabled": True},
    max_tokens=4096,
    temperature=0.1
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Open-source with academic focus
- More verbose reasoning output
- Good for educational applications
- Validate mathematical/logical accuracy

---

## OpenAI GPT-OSS 120B

**Model ID**: `openai/gpt-oss-120b:free`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Free tier
- Use Case: Long-context analysis, document processing, complex reasoning

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:free",
    messages=[{"role": "user", "content": "Analyze this research paper..."}],
    reasoning={"enabled": True},
    max_tokens=8192,
    temperature=0.3
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Leverage 128K context for document analysis
- Consider chunking near limits
- Good general-purpose depth

---

## OpenAI GPT-OSS 20B

**Model ID**: `openai/gpt-oss-20b:free`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Architecture: Mixture-of-Experts (MoE, 3.6B active params)
- Cost: Free tier
- Use Case: Agentic capabilities, structured reasoning, function calling

**Request Schema (Multi-Turn Reasoning)**:
```python
# First call
response = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=[{"role": "user", "content": "How many r's in 'strawberry'?"}],
    extra_body={"reasoning": {"enabled": True}},
    max_tokens=2048,
    temperature=0.3
)

# Preserve reasoning_details for follow-up
messages = [
    {"role": "user", "content": "How many r's in 'strawberry'?"},
    {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "reasoning_details": response.choices[0].message.reasoning_details
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second call continues reasoning
response2 = client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=messages,
    extra_body={"reasoning": {"enabled": True}}
)
```

**Agent Notes**:
- MoE architecture optimized for low-latency
- Supports Harmony response format
- Function calling & structured outputs
- Good balance for agentic workflows

---

## TNG DeepSeek R1T2 Chimera

**Model ID**: `tngtech/deepseek-r1t2-chimera:free`

**Specifications**:
- Context: 163,840 tokens (tested to ~130K)
- Parameters: 671B (MoE)
- Modality: Text → Text
- Cost: Free tier
- Use Case: Long-context dialogue, open-ended generation, reasoning

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="tngtech/deepseek-r1t2-chimera:free",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    extra_headers={
        "HTTP-Referer": "YOUR_SITE_URL",
        "X-Title": "YOUR_SITE_NAME"
    },
    max_tokens=4096,
    temperature=0.7
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Consistent `<think>` token behavior
- 20% faster than R1, 2× faster than R1-0528
- Excellent cost-to-intelligence ratio
- Monitor context near 130K token limits

---

## Z.AI GLM 4.5 Air

**Model ID**: `z-ai/glm-4.5-air:free`

**Specifications**:
- Context: 131,072 tokens
- Architecture: Compact MoE
- Modality: Text → Text
- Cost: Free tier
- Use Case: Agent applications, real-time interaction, tool use

**Request Schema (Hybrid Modes)**:
```python
# Thinking mode (advanced reasoning + tool use)
completion = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=[{"role": "user", "content": "Solve this problem with tools."}],
    extra_body={"reasoning": {"enabled": True}},
    extra_headers={"HTTP-Referer": "SITE_URL", "X-Title": "SITE_NAME"},
    max_tokens=2048,
    temperature=0.7
)

# Non-thinking mode (fast real-time responses)
completion = client.chat.completions.create(
    model="z-ai/glm-4.5-air:free",
    messages=[{"role": "user", "content": "Quick response needed."}],
    max_tokens=1024,
    temperature=0.7
)
```

**Agent Notes**:
- Choose mode: thinking (complex) vs non-thinking (speed)
- Purpose-built for agent-centric apps
- Excellent for tool integration
- Compact MoE design balances performance/capability

---

## NVIDIA Nemotron Nano 9B v2

**Model ID**: `nvidia/nemotron-nano-9b-v2:free`

**Specifications**:
- Context: 8,192 tokens
- Modality: Text → Text
- Cost: Free tier
- Use Case: Quick responses, lightweight tasks, conversational AI

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="nvidia/nemotron-nano-9b-v2:free",
    messages=[{"role": "user", "content": "Explain quantum physics simply."}],
    max_tokens=1024,
    temperature=0.7,
    extra_headers={"HTTP-Referer": "SITE_URL", "X-Title": "SITE_NAME"}
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Fast response times (interactive apps)
- Limited 8K context window
- Not ideal for deep analytical tasks

---

## Meituan LongCat Flash Chat

**Model ID**: `meituan/longcat-flash-chat:free`

**Specifications**:
- Context: 32,768 tokens
- Modality: Text → Text
- Cost: Free tier
- Use Case: Balanced general conversation, moderate complexity

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="meituan/longcat-flash-chat:free",
    messages=[{"role": "user", "content": "Write a creative story."}],
    max_tokens=2048,
    temperature=0.8,
    extra_headers={"HTTP-Referer": "SITE_URL", "X-Title": "SITE_NAME"}
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- Good all-purpose reliability
- Balanced speed/context (32K)
- Suitable for creative and analytical work

---

## Google Gemma 3 27B IT

**Model ID**: `google/gemma-3-27b-it:free`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text + Image → Text
- Cost: Free tier
- Use Case: Vision-language, multilingual, structured outputs

**Request Schema (Multimodal)**:
```python
# Text-only
completion = client.chat.completions.create(
    model="google/gemma-3-27b-it:free",
    messages=[{"role": "user", "content": "Explain quantum computing."}],
    max_tokens=2048,
    temperature=0.3
)

# Text + Image
completion = client.chat.completions.create(
    model="google/gemma-3-27b-it:free",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    extra_headers={"HTTP-Referer": "SITE_URL", "X-Title": "SITE_NAME"}
)

result = completion.choices[0].message.content
```

**Agent Notes**:
- **Critical**: Mixed content MUST be array of typed objects, not string
- Always validate `choices` exists before parsing
- Handle null responses from OpenRouter
- Validate image URLs before requests
- Excellent for vision-language & multilingual tasks

---

# NVIDIA NIM Provider

## Meta Llama 3.1 405B Instruct

**Model ID**: `meta/llama-3.1-405b-instruct`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Paid API access
- Use Case: High-accuracy research, complex reasoning

**Client Initialization**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="YOUR_NVIDIA_API_KEY"
)
```

**Request Schema (Streaming)**:
```python
completion = client.chat.completions.create(
    model="meta/llama-3.1-405b-instruct",
    messages=[{"role": "user", "content": "Detailed analysis..."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=4096,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**Agent Notes**:
- Highest accuracy, highest cost
- Handle streaming for large outputs
- Best for high-stakes applications

---

## Meta Llama 3.1 70B Instruct

**Model ID**: `meta/llama-3.1-70b-instruct`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Paid API access
- Use Case: High-performance general tasks, content generation

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Generate comprehensive content..."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=2048,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**Agent Notes**:
- Excellent general-purpose model
- Good balance: performance vs cost
- Ideal for professional applications

---

## Meta Llama 3.1 8B Instruct

**Model ID**: `meta/llama-3.1-8b-instruct`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Paid API access
- Use Case: Fast responses, lightweight processing

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Quick analysis..."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**Agent Notes**:
- Best speed-to-cost ratio
- Good for simple to moderate tasks
- Consider upgrading for complex reasoning

---

## MistralAI Mistral Nemotron

**Model ID**: `mistralai/mistral-nemotron`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Paid API access
- Use Case: Balanced reasoning, educational applications

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="mistralai/mistral-nemotron",
    messages=[{"role": "user", "content": "Reason through this problem..."}],
    temperature=0.6,  # Higher default for creativity
    top_p=0.7,
    max_tokens=4096,  # Larger default
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**Agent Notes**:
- Higher temperature (0.6) produces more creative outputs
- Larger token limits (4096) for detailed responses
- Good for analytical and educational tasks

---

## Qwen Qwen2.5 7B Instruct

**Model ID**: `qwen/qwen2.5-7b-instruct`

**Specifications**:
- Context: 131,072 tokens
- Modality: Text → Text
- Cost: Paid API access
- Use Case: Multilingual applications, efficient general tasks

**Request Schema**:
```python
completion = client.chat.completions.create(
    model="qwen/qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "回答这个问题..."}],  # Multilingual support
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

**Agent Notes**:
- Excellent multilingual capabilities
- Efficient 7B parameter model
- Good balance: speed vs capability
- Ideal for international projects

---

## Implementation Guidelines

### Model Selection Matrix

| Use Case | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Lightweight Tasks** | Nemotron Nano 9B, LongCat Flash | Llama 3.1 8B |
| **Reasoning Tasks** | Tongyi DeepResearch, OLMo 3, DeepSeek Chimera | Llama 3.1 70B, Mistral Nemotron |
| **Multimodal** | Gemma 3 27B | N/A |
| **Long Context** | GPT-OSS 120B, DeepSeek Chimera | Llama 3.1 405B |
| **Multilingual** | Gemma 3 27B | Qwen 2.5 7B |
| **Agentic Workflows** | GLM 4.5 Air, GPT-OSS 20B | Llama 3.1 70B |

### Best Practices

**Error Handling**:
```python
try:
    completion = client.chat.completions.create(...)
    if completion.choices and len(completion.choices) > 0:
        result = completion.choices[0].message.content
    else:
        result = None  # Handle null response
except Exception as e:
    # Log error, implement retry logic
    pass
```

**Rate Limiting**:
- Implement exponential backoff for retries
- Track API usage and costs
- Use free tiers for development, paid for production

**Context Management**:
- Track token usage per request
- Chunk large inputs to stay under limits
- Preserve reasoning context in multi-turn conversations

**Streaming**:
```python
for chunk in completion:
    if chunk.choices[0].delta.content:
        # Stream to user, buffer for logging
        print(chunk.choices[0].delta.content, end="")
```

**Cost Optimization**:
- Use free tiers for prototyping
- Upgrade to paid tiers only when accuracy demands it
- Choose smallest model that meets requirements
- Monitor usage patterns and optimize model selection

---

**Document Maintenance**: Update model IDs, context limits, and pricing as providers release changes.
