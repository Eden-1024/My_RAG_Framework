import os
from openai import OpenAI
import json
import random
import re
import math
import difflib
from collections import Counter
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 从 JSON 文件读取自然语言查询，默认文件位于与当前脚本同目录下的 query.json
json_path = '../data/q2sql_pairs.json'

try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 从读取到的 data 中随机选取一个条目，并尝试提取问题和答案字段
    selected = None
    question = None
    answer = None

    if isinstance(data, list) and data:
        selected = random.choice(data)
        question = selected.get("question") or selected.get("query") or selected.get("nl_query")
        answer = selected.get("sql") or selected.get("sql_query")
    else:
        raise ValueError("JSON 文件内容不是预期的列表格式或为空") 

    user_query = str(question)
    print(f"随机选择的问题：{user_query}")
    if answer:
        print(f"对应的答案：{answer}")
except Exception as e:
    user_query = ""
    print(f"无法读取 JSON 文件 ({json_path})：{e}")
    exit
    

# 准备生成SQL的提示词
prompt = f"""
用户的自然语言问题如下：
"{user_query}"
请注意：
请只返回SQL查询语句，不要包含任何其他解释、注释或格式标记（如```sql）
"""

# print(f"\n生成SQL的提示词：\n{prompt}")

# 调用LLM生成SQL语句
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个SQL专家。请只返回SQL查询语句，不要包含任何Markdown格式或其他说明。"},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)
# print(f"\n完整的LLM响应：\n{response}")
# 清理SQL语句，移除可能的Markdown标记
sql = response.choices[0].message.content.strip()
# sql = sql.replace('```sql', '').replace('```', '').strip()
print(f"\n生成的SQL查询语句：\n{sql}")


def normalize_sql(s: str) -> str:
    if not s:
        return ""
    # remove possible markdown code fences and SQL comments, lowercase, collapse whitespace
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    s = re.sub(r"--.*?$", "", s, flags=re.M)      # single-line comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)   # block comments
    s = s.replace("`", "").replace('"', "").replace(";", " ")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_sql(s: str):
    # simple SQL tokenizer: words, numbers, operators, parentheses, commas, dots
    if not s:
        return []
    return re.findall(r"\w+|\S", s)

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens)>=n else []

def modified_precision(candidate_tokens, reference_tokens, n):
    cand_ngrams = ngrams(candidate_tokens, n)
    ref_ngrams = ngrams(reference_tokens, n)
    if not cand_ngrams:
        return 0.0, 0, 0
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)
    overlap = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
    return overlap / len(cand_ngrams), overlap, len(cand_ngrams)

def bleu(candidate_tokens, reference_tokens, max_n=4):
    precisions = []
    matches = []
    totals = []
    for n in range(1, max_n+1):
        p, m, t = modified_precision(candidate_tokens, reference_tokens, n)
        precisions.append(p if p > 0 else 1e-16)  # avoid log(0)
        matches.append(m)
        totals.append(t)
    # geometric mean of precisions
    log_prec_sum = sum((1.0/max_n) * math.log(p) for p in precisions)
    geo_mean = math.exp(log_prec_sum)
    c = len(candidate_tokens)
    r = len(reference_tokens)
    bp = 1.0 if c > r else math.exp(1 - r/c) if c>0 else 0.0
    score = bp * geo_mean
    return score, {"matches": matches, "totals": totals, "bp": bp}

def lcs_length(a, b):
    # classic DP for LCS length
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    dp = [0] * (lb + 1)
    for i in range(la-1, -1, -1):
        newdp = [0] * (lb + 1)
        for j in range(lb-1, -1, -1):
            if a[i] == b[j]:
                newdp[j] = 1 + dp[j+1]
            else:
                newdp[j] = max(dp[j], newdp[j+1])
        dp = newdp
    return dp[0]

def rouge_l(candidate_tokens, reference_tokens):
    lcs = lcs_length(candidate_tokens, reference_tokens)
    cand_len = len(candidate_tokens)
    ref_len = len(reference_tokens)
    prec = lcs / cand_len if cand_len else 0.0
    rec = lcs / ref_len if ref_len else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = (2 * prec * rec) / (prec + rec)
    return {"lcs": lcs, "precision": prec, "recall": rec, "f1": f1}

def token_overlap(candidate_tokens, reference_tokens):
    cand_counts = Counter(candidate_tokens)
    ref_counts = Counter(reference_tokens)
    overlap = sum(min(cand_counts[t], ref_counts[t]) for t in cand_counts)
    prec = overlap / sum(cand_counts.values()) if cand_counts else 0.0
    rec = overlap / sum(ref_counts.values()) if ref_counts else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return {"overlap": overlap, "precision": prec, "recall": rec, "f1": f1}

def evaluate_sql(candidate_sql: str, reference_sql: str):
    cand_norm = normalize_sql(candidate_sql)
    ref_norm = normalize_sql(reference_sql)
    cand_tokens = tokenize_sql(cand_norm)
    ref_tokens = tokenize_sql(ref_norm)

    results = {}
    results["candidate_normalized"] = cand_norm
    results["reference_normalized"] = ref_norm
    results["exact_match"] = cand_norm == ref_norm
    results["sequence_ratio"] = difflib.SequenceMatcher(None, cand_norm, ref_norm).ratio()
    results["token_count"] = {"candidate": len(cand_tokens), "reference": len(ref_tokens)}
    results["token_overlap"] = token_overlap(cand_tokens, ref_tokens)
    bleu_score, bleu_info = bleu(cand_tokens, ref_tokens, max_n=4)
    results["bleu"] = {"score": bleu_score, **bleu_info}
    results["rouge_l"] = rouge_l(cand_tokens, ref_tokens)
    return results

# prepare inputs (answer may be None or already a dict/list)
reference_sql = answer if isinstance(answer, str) else (json.dumps(answer, ensure_ascii=False) if answer is not None else "")
candidate_sql = sql if isinstance(sql, str) else (json.dumps(sql, ensure_ascii=False) if sql is not None else "")

metrics = evaluate_sql(candidate_sql, reference_sql)

# print a concise summary
print("\n评价指标（越高越好，部分为比率/分数）:")
print(f"- Exact match: {metrics['exact_match']}")
print(f"- Sequence similarity (ratio): {metrics['sequence_ratio']:.4f}")
print(f"- BLEU-4 (BP * geom mean): {metrics['bleu']['score']:.4f}  (bp={metrics['bleu']['bp']:.4f})")
print(f"- ROUGE-L F1: {metrics['rouge_l']['f1']:.4f}  (LCS={metrics['rouge_l']['lcs']})")
to = metrics["token_overlap"]
print(f"- Token Overlap P/R/F1: {to['precision']:.4f} / {to['recall']:.4f} / {to['f1']:.4f}  (overlap={to['overlap']})")
print(f"- Candidate tokens: {metrics['token_count']['candidate']}, Reference tokens: {metrics['token_count']['reference']}")

# optionally print normalized SQLs for inspection
print("\n标准化后候选 SQL：")
print(metrics["candidate_normalized"] or "(空)")

print("\n标准化后参考 SQL：")
print(metrics["reference_normalized"] or "(空)")