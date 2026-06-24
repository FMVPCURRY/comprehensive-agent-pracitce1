"""
reconstructed_fraud_crawler.py

中文网络欺诈文本采集脚本（根据 ChiFraud 数据形态反推设计）

用途：
    1. 从公开网页搜索结果中采集中文网页标题、摘要和正文片段。
    2. 使用欺诈类别关键词对文本进行弱标注。
    3. 将多分类标签转换为二分类标签：
           label = 0  正常文本
           label = 1  网络欺诈/黑灰产相关文本
    4. 对手机号、URL、QQ、微信号等敏感信息进行脱敏。
    5. 导出 text,label 格式的 CSV 文件，用于网络欺诈检测模型训练。

重要说明：
    该脚本不是 ChiFraud 官方原始爬虫，而是根据 ChiFraud 的数据字段、
    类别设计和网页短文本形态，反推出的一套可复现的数据采集流程。

依赖安装：
    pip install requests beautifulsoup4 pandas lxml

运行示例：
    python reconstructed_fraud_crawler.py --output fraud_crawled_text_label.csv --max-pages 100

合规建议：
    - 只采集公开网页。
    - 遵守网站 robots.txt 和服务条款。
    - 不采集私信、群聊、登录后内容。
    - 入库前必须脱敏。
    - 弱标注数据建议人工抽检后再用于正式训练。
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Iterable
from urllib.parse import quote_plus, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
}


# 参考 ChiFraud 原始类别设计：
# 0 正常
# 1 赌博博彩
# 2 招嫖色情
# 3 办假证
# 4 虚假办卡
# 5 违禁药品交易
# 6 违规提现
# 7 虚假证明
# 8 虚假手机卡
# 9 地下黑贷
# 10 新类型
FRAUD_CATEGORY_KEYWORDS = {
    1: {
        "category": "赌博博彩",
        "queries": [
            "大小单双 稳赚 加微信",
            "时时彩 计划导师 QQ",
            "北京赛车 计划群 微信",
            "棋牌代理 返水 招募",
            "彩票平台 邀请码 赚钱",
        ],
        "keywords": ["博彩", "开奖", "下注", "精准计划", "导师", "返水", "邀请码", "送彩金"],
    },
    2: {
        "category": "招嫖色情",
        "queries": [
            "同城上门 服务 微信",
            "外围 模特 预约 QQ",
            "桑拿会所 上门 服务",
            "兼职妹妹 上门 微信",
        ],
        "keywords": ["上门", "服务", "会所", "外围", "预约", "技师", "包夜", "全套"],
    },
    3: {
        "category": "办假证",
        "queries": [
            "办理毕业证 成绩单 微信",
            "代办学历认证 QQ",
            "办证 身份证 驾驶证 微信",
            "国外大学文凭 代办",
        ],
        "keywords": ["代办", "毕业证", "学历认证", "成绩单", "驾驶证", "文凭", "证书"],
    },
    4: {
        "category": "虚假办卡",
        "queries": [
            "办理银行卡 信用卡 黑户",
            "信用卡代办 包过 微信",
            "大额信用卡 包下卡",
            "银行卡 四件套 出售",
        ],
        "keywords": ["办卡", "信用卡", "包过", "下卡", "银行卡", "四件套"],
    },
    5: {
        "category": "违禁药品交易",
        "queries": [
            "违禁药品 货到付款 QQ",
            "迷药 出售 微信",
            "精神药品 渠道 购买",
            "管制药品 出售",
        ],
        "keywords": ["药品", "迷药", "管制", "货到付款", "渠道", "出售"],
    },
    6: {
        "category": "违规提现",
        "queries": [
            "信用卡套现 花呗取现 微信",
            "白条套现 秒到",
            "信用卡代还 垫还",
            "支付宝花呗 提现",
        ],
        "keywords": ["套现", "取现", "提现", "代还", "垫还", "秒到", "花呗", "白条"],
    },
    7: {
        "category": "虚假证明",
        "queries": [
            "医院诊断证明 代开",
            "病假条 代开 微信",
            "收入证明 工作证明 代办",
            "银行流水 证明 制作",
        ],
        "keywords": ["证明", "代开", "病假条", "诊断证明", "收入证明", "银行流水"],
    },
    8: {
        "category": "虚假手机卡",
        "queries": [
            "不用实名 手机卡 出售",
            "物联网卡 批发 不实名",
            "手机卡 注册账号 接码",
            "短信验证码 接码平台",
        ],
        "keywords": ["手机卡", "不用实名", "接码", "验证码", "注册账号", "物联网卡"],
    },
    9: {
        "category": "地下黑贷",
        "queries": [
            "黑户贷款 包装流水",
            "无视征信 贷款 当天放款",
            "贷款包装 技术 加微信",
            "空放 私人借贷",
        ],
        "keywords": ["黑户", "贷款", "包装", "无视征信", "当天放款", "空放", "私人借贷"],
    },
    10: {
        "category": "新类型",
        "queries": [
            "刷单返利 兼职 日结",
            "杀猪盘 投资 群聊",
            "客服退款 验证码 转账",
            "虚拟币搬砖 稳赚",
            "账号解冻 保证金",
        ],
        "keywords": ["刷单", "返利", "杀猪盘", "退款", "验证码", "虚拟币", "解冻", "保证金"],
    },
}


NORMAL_QUERIES = [
    "银行 数字化 转型 新闻",
    "农业 技术 推广 文章",
    "旅游 攻略 中文 博客",
    "学习经验 分享 中文",
    "电脑维修 教程 中文",
    "电影评论 中文",
    "招聘面试经验 分享",
    "普通商品 介绍 电商",
    "健康科普 常识",
    "学校招生 通知",
]


@dataclass
class CrawledRecord:
    text: str
    label_id: int
    label: int
    category: str
    source_url: str
    source_domain: str
    query: str
    text_hash: str


class FraudTextCrawler:
    def __init__(
        self,
        max_pages: int = 100,
        timeout: int = 12,
        min_text_len: int = 20,
        max_text_len: int = 800,
        sleep_min: float = 1.0,
        sleep_max: float = 3.0,
    ) -> None:
        self.max_pages = max_pages
        self.timeout = timeout
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def run(self) -> list[CrawledRecord]:
        records: list[CrawledRecord] = []

        for label_id, config in FRAUD_CATEGORY_KEYWORDS.items():
            for query in config["queries"]:
                urls = self.search_urls(query)
                for url in urls:
                    text = self.fetch_and_extract(url)
                    if not text:
                        continue
                    record = self.build_record(
                        text=text,
                        label_id=label_id,
                        category=config["category"],
                        source_url=url,
                        query=query,
                    )
                    records.append(record)
                    self.polite_sleep()
                    if len(records) >= self.max_pages:
                        return self.deduplicate(records)

        for query in NORMAL_QUERIES:
            urls = self.search_urls(query)
            for url in urls:
                text = self.fetch_and_extract(url)
                if not text:
                    continue
                record = self.build_record(
                    text=text,
                    label_id=0,
                    category="正常",
                    source_url=url,
                    query=query,
                )
                records.append(record)
                self.polite_sleep()
                if len(records) >= self.max_pages:
                    return self.deduplicate(records)

        return self.deduplicate(records)

    def search_urls(self, query: str, limit: int = 10) -> list[str]:
        """
        使用搜索引擎 HTML 页面获取候选 URL。

        这里用 DuckDuckGo 的 HTML 搜索页做示例。实际项目中也可以替换为：
            - 合规搜索 API
            - 自建种子 URL 列表
            - 论坛公开列表页
            - 官方反诈案例站点
        """
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query + ' 中文')}"
        try:
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        urls: list[str] = []
        for a_tag in soup.select("a.result__a"):
            href = a_tag.get("href", "").strip()
            if href.startswith("http") and self.is_allowed_url(href):
                urls.append(href)
            if len(urls) >= limit:
                break
        return urls

    def fetch_and_extract(self, url: str) -> str | None:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None

        response.encoding = response.apparent_encoding or response.encoding
        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        description = ""
        meta_desc = soup.find("meta", attrs={"name": re.compile("description", re.I)})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"]

        body_candidates = []
        for selector in ["article", "main", ".content", ".article", ".post", "body"]:
            node = soup.select_one(selector)
            if node:
                body_candidates.append(node.get_text(" ", strip=True))

        raw_text = " ".join([title, description] + body_candidates)
        text = self.clean_text(raw_text)

        if len(text) < self.min_text_len:
            return None
        return text[: self.max_text_len]

    def build_record(
        self,
        text: str,
        label_id: int,
        category: str,
        source_url: str,
        query: str,
    ) -> CrawledRecord:
        label = 0 if label_id == 0 else 1
        source_domain = urlparse(source_url).netloc
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return CrawledRecord(
            text=text,
            label_id=label_id,
            label=label,
            category=category,
            source_url=source_url,
            source_domain=source_domain,
            query=query,
            text_hash=text_hash,
        )

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text, flags=re.I)
        text = re.sub(r"\b1[3-9]\d{9}\b", "[PHONE]", text)
        text = re.sub(r"\b\d{5,12}\b", "[NUM]", text)
        text = re.sub(
            r"(微信|vx|v信|qq|QQ|Q|加q|加Q|WeChat)[:：]?\s*[A-Za-z0-9_\-]{5,}",
            r"\1[CONTACT]",
            text,
        )
        text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
        return text

    def deduplicate(self, records: Iterable[CrawledRecord]) -> list[CrawledRecord]:
        seen = set()
        result = []
        for record in records:
            if record.text_hash in seen:
                continue
            seen.add(record.text_hash)
            result.append(record)
        return result

    def is_allowed_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        blocked_ext = (".jpg", ".png", ".gif", ".zip", ".rar", ".pdf", ".doc", ".docx")
        if parsed.path.lower().endswith(blocked_ext):
            return False
        return True

    def polite_sleep(self) -> None:
        time.sleep(random.uniform(self.sleep_min, self.sleep_max))


def save_records(records: list[CrawledRecord], output_path: str) -> None:
    rows = [asdict(record) for record in records]
    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "text",
                "label_id",
                "label",
                "category",
                "source_url",
                "source_domain",
                "query",
                "text_hash",
            ]
        )

    # 训练时最常用的是 text,label 两列；
    # 其他字段用于追踪来源、复核标签和写实验说明。
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    simple_output = output_path.replace(".csv", "_text_label.csv")
    df[["text", "label"]].to_csv(simple_output, index=False, encoding="utf-8-sig")

    print(f"完整采集文件: {output_path}")
    print(f"训练用二分类文件: {simple_output}")
    print(f"样本数量: {len(df)}")
    if not df.empty:
        print("标签分布:")
        print(df["label"].value_counts().sort_index())
        print("类别分布:")
        print(df["category"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文网络欺诈文本采集与弱标注脚本")
    parser.add_argument("--output", default="fraud_crawled_records.csv", help="输出 CSV 文件路径")
    parser.add_argument("--max-pages", type=int, default=100, help="最多采集网页数量")
    parser.add_argument("--min-text-len", type=int, default=20, help="最短文本长度")
    parser.add_argument("--max-text-len", type=int, default=800, help="最长文本截断长度")
    parser.add_argument("--sleep-min", type=float, default=1.0, help="最小请求间隔秒数")
    parser.add_argument("--sleep-max", type=float, default=3.0, help="最大请求间隔秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawler = FraudTextCrawler(
        max_pages=args.max_pages,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )
    records = crawler.run()
    save_records(records, args.output)


if __name__ == "__main__":
    main()
