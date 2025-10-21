"""
News Intelligence Service
- Fetches news from ET, Moneycontrol, Business Standard via RSS endpoints
- Uses Claude to extract mentioned stocks and sentiment per article
- Validates symbols with yfinance
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import re
import json

import requests
import yfinance as yf

from app.services.claude_client import claude_client
from app.services.stocks import StocksService

logger = logging.getLogger(__name__)


class NewsIntelligenceService:
    def __init__(self):
        # RSS feeds (public) for Economic Times, Moneycontrol, Business Standard
        # These feeds may change; handle failures gracefully
        self.feeds = [
            # Economic Times Markets RSS
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            # Moneycontrol and Business Standard can return 403; keep but tolerate failures
            "https://www.moneycontrol.com/rss/MCtopnews.xml",
            "https://www.business-standard.com/rss/markets-106.rss",
        ]
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NewsIntelligenceBot/1.0 (+https://example.com)"
        })

        # Allowed NSE universe to reduce noise and false-positives
        try:
            self._stocks_service = StocksService()
            self.allowed_symbols = set(self._stocks_service.get_universe_symbols(limit=200))
        except Exception:
            self._stocks_service = None
            self.allowed_symbols = set()

        # Common words to ignore as symbols
        self.blacklist_words = {
            "AND", "THE", "MARKET", "BANK", "SENSEX", "NIFTY", "CDATA",
            "EARNINGS", "RALLY", "JUMPS", "FLOWS", "SEASON", "PREMIUM",
            "PTS", "SIP", "DOMESTIC", "DECADE"
        }

    def _fetch_feed_entries(self, limit_per_feed: int = 30) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for url in self.feeds:
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code != 200:
                    logger.warning(f"Feed fetch failed {url}: {resp.status_code}")
                    continue
                # Simple RSS parsing via regex fallbacks to avoid extra deps
                # Extract <item>...</item> blocks
                items = re.findall(r"<item>([\s\S]*?)</item>", resp.text, re.IGNORECASE)
                for item in items[:limit_per_feed]:
                    title_match = re.search(r"<title>([\s\S]*?)</title>", item, re.IGNORECASE)
                    desc_match = re.search(r"<description>([\s\S]*?)</description>", item, re.IGNORECASE)
                    link_match = re.search(r"<link>([\s\S]*?)</link>", item, re.IGNORECASE)
                    title = (title_match.group(1) if title_match else "").strip()
                    description = (re.sub(r"<.*?>", " ", desc_match.group(1)) if desc_match else "").strip()
                    link = (link_match.group(1) if link_match else "").strip()
                    if not title:
                        continue
                    entries.append({
                        "title": title,
                        "description": description,
                        "link": link,
                        "source": url
                    })
            except Exception as e:
                logger.warning(f"Failed fetching feed {url}: {e}")
            time.sleep(0.2)
        return entries

    def _call_claude_extract(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not claude_client.client:
            logger.error("Claude client not initialized")
            return []
        try:
            allowed_list = sorted(list(self.allowed_symbols)) if self.allowed_symbols else []
            system = (
                "You are an equity news analyst. Return ONLY JSON (no prose, no code fences).\n"
                "Schema: {\"results\":[{\"symbol\":string,\"name\"?:string,\"mentions\":int,\"sentiment\":float,\"sources\":[string],\"excerpt\"?:string}]}\n"
                "Rules:\n- Use ONLY NSE symbols from ALLOWED_SYMBOLS below.\n- Output plain symbol without any prefix like '$' or exchange suffix.\n- Omit uncertain tickers.\n- Sentiment: positive>0, negative<0, neutral≈0.\n- Aggregate mentions across articles.\n"
                f"ALLOWED_SYMBOLS: {json.dumps(allowed_list)}"
            )
            user_payload = {
                "articles": [
                    {
                        "title": a.get("title"),
                        "summary": a.get("description", ""),
                        "source": a.get("source"),
                        "link": a.get("link"),
                    }
                    for a in articles
                ]
            }
            user = (
                "Analyze these Indian market news items and extract symbols with sentiment. Respond with JSON only.\n\n" +
                json.dumps(user_payload, ensure_ascii=False)
            )
            response = claude_client.client.messages.create(
                model=claude_client.model,
                max_tokens=min(claude_client.max_tokens, 1200),
                temperature=0.1,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text.strip()
            # Lenient JSON extraction: strip code fences and parse dict or list
            text = re.sub(r"^```[a-zA-Z]*|```$", "", text).strip()
            start = text.find("{")
            end = text.rfind("}")
            parsed: Any
            if start != -1 and end != -1:
                parsed = json.loads(text[start : end + 1])
            else:
                # Try list form
                lstart = text.find("[")
                lend = text.rfind("]")
                if lstart != -1 and lend != -1:
                    parsed = {"results": json.loads(text[lstart : lend + 1])}
                else:
                    return []
            results = parsed.get("results", [])
            if isinstance(results, list):
                return results
            return []
        except Exception as e:
            logger.error(f"Claude extract failed: {e}")
            return []

    def _naive_extract_symbols(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback extractor: find ALLCAPS tokens that look like symbols, validate via yfinance, aggregate sentiment as neutral."""
        token_re = re.compile(r"\b[A-Z][A-Z0-9&.]{2,15}\b")
        counts: Dict[str, Dict[str, Any]] = {}
        for a in articles:
            text = f"{a.get('title','')} {a.get('description','')}"
            candidates = set(token_re.findall(text.upper()))
            for cand in candidates:
                symbol = self._normalize_symbol(cand)
                if not self._is_valid_symbol(symbol):
                    continue
                if symbol not in counts:
                    counts[symbol] = {
                        "symbol": symbol,
                        "name": None,
                        "mentions": 0,
                        "sentiment": 0.0,
                        "sources": set(),
                        "excerpt": a.get("description") or a.get("title"),
                    }
                counts[symbol]["mentions"] += 1
                counts[symbol]["sources"].add(a.get("source") or "unknown")
        results: List[Dict[str, Any]] = []
        for sym, data in counts.items():
            results.append({
                "symbol": sym,
                "name": data.get("name"),
                "mentions": data["mentions"],
                "sentiment": 0.0,
                "sources": sorted(list(data["sources"])),
                "excerpt": data.get("excerpt"),
            })
        return results

    def _deterministic_extract_symbols(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract symbols by scanning text for allowed NSE symbols as whole words only."""
        if not self.allowed_symbols:
            return []
        # Bucket symbols by first letter for small regexes
        buckets: Dict[str, List[str]] = {}
        for sym in self.allowed_symbols:
            if not sym:
                continue
            buckets.setdefault(sym[0], []).append(sym)

        extracted: Dict[str, Dict[str, Any]] = {}
        for a in articles:
            text = f"{a.get('title','')}\n{a.get('description','')}".upper()
            for first, syms in buckets.items():
                if first not in text:
                    continue
                pattern = r"\\b(" + "|".join(map(re.escape, syms)) + r")\\b"
                for match in re.findall(pattern, text):
                    symbol = self._normalize_symbol(match)
                    if not symbol:
                        continue
                    if symbol not in extracted:
                        extracted[symbol] = {
                            "symbol": symbol,
                            "name": None,
                            "mentions": 0,
                            "sentiment": 0.0,
                            "sources": set(),
                            "excerpt": a.get("description") or a.get("title"),
                            "_texts": []
                        }
                    extracted[symbol]["mentions"] += 1
                    extracted[symbol]["sources"].add(a.get("source") or "unknown")
                    extracted[symbol]["_texts"].append(text[:500])

        return [
            {
                "symbol": s,
                "name": d.get("name"),
                "mentions": d["mentions"],
                "sentiment": 0.0,
                "sources": sorted(list(d["sources"])),
                "excerpt": d.get("excerpt"),
                "_texts": d["_texts"],
            }
            for s, d in extracted.items()
        ]

    def _claude_score_sentiment(self, symbol_to_texts: Dict[str, List[str]]) -> Dict[str, float]:
        """Use Claude to score sentiment per symbol with a constrained prompt. Returns symbol->sentiment."""
        if not claude_client.client:
            return {}
        try:
            items = [
                {"symbol": sym, "texts": texts[:5]}
                for sym, texts in symbol_to_texts.items()
            ]
            system = (
                'You are an equity sentiment scorer. Return ONLY JSON.\n'
                'Format: {"scores": [{"symbol": "SYM", "sentiment": 0.0}]}\n'
                'Sentiment range: -1.0 (very negative) to 1.0 (very positive). Neutral≈0.'
            )
            user = json.dumps({"items": items}, ensure_ascii=False)
            response = claude_client.client.messages.create(
                model=claude_client.model,
                max_tokens=min(claude_client.max_tokens, 800),
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text.strip()
            text = re.sub(r"^```[a-zA-Z]*|```$", "", text).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return {}
            parsed = json.loads(text[start:end+1])
            scores = parsed.get("scores", [])
            out: Dict[str, float] = {}
            for s in scores:
                sym = str(s.get("symbol", "")).upper()
                try:
                    out[sym] = float(s.get("sentiment", 0.0))
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.error(f"Claude sentiment scoring failed: {e}")
            return {}

    def _is_valid_symbol(self, symbol: str) -> bool:
        try:
            clean = symbol.replace('.NS', '')
            if self.allowed_symbols and clean not in self.allowed_symbols:
                return False
            # Try NSE suffix first; support raw symbol too
            candidates = [clean, f"{clean}.NS"]
            for ticker_symbol in candidates:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.fast_info if hasattr(ticker, "fast_info") else {}
                if info and getattr(info, "last_price", None) or info.get("last_price"):
                    return True
                # Fallback to history check
                hist = ticker.history(period="5d")
                if hist is not None and not hist.empty:
                    return True
            return False
        except Exception:
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        # Strip leading '$' and punctuation
        s = re.sub(r"^[^A-Z0-9]+", "", s)
        s = re.sub(r"[^A-Z0-9.]+$", "", s)
        # Discard blacklist words
        if s in self.blacklist_words:
            return ""
        if not s.endswith(".NS"):
            return s  # Keep base; frontend/backends may add suffix consistently elsewhere
        return s

    def analyze_news(self, max_results: int = 20) -> List[Dict[str, Any]]:
        articles = self._fetch_feed_entries(limit_per_feed=30)
        if not articles:
            return []
        # Prefer deterministic extraction to avoid wrong symbols
        raw_results = self._deterministic_extract_symbols(articles)
        if not raw_results:
            # Fallback to Claude extraction, then naive
            raw_results = self._call_claude_extract(articles)
            if not raw_results:
                raw_results = self._naive_extract_symbols(articles)

        # Collect texts for sentiment per symbol
        symbol_texts: Dict[str, List[str]] = {}
        for item in raw_results:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            symbol = self._normalize_symbol(symbol)
            if not symbol:
                continue
            if "_texts" in item:
                symbol_texts.setdefault(symbol, []).extend(item["_texts"])

        sentiments: Dict[str, float] = {}
        if symbol_texts:
            sentiments = self._claude_score_sentiment(symbol_texts)

        # Aggregate
        aggregated: Dict[str, Dict[str, Any]] = {}
        for item in raw_results:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            symbol = self._normalize_symbol(symbol)
            if not symbol:
                continue
            sentiment = float(sentiments.get(symbol, item.get("sentiment", 0.0)))
            mentions = int(item.get("mentions", 1))
            sources = item.get("sources", []) or []
            name = item.get("name")
            excerpt = item.get("excerpt")

            # Validate via yfinance
            if not self._is_valid_symbol(symbol):
                continue

            if symbol not in aggregated:
                aggregated[symbol] = {
                    "symbol": symbol,
                    "name": name,
                    "mentions": 0,
                    "sentiment_sum": 0.0,
                    "sources": set(),
                    "excerpt": excerpt,
                }
            aggregated[symbol]["mentions"] += max(1, mentions)
            aggregated[symbol]["sentiment_sum"] += sentiment
            for src in sources:
                aggregated[symbol]["sources"].add(src)

        # Score: mentions weight + avg sentiment bonus
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for sym, data in aggregated.items():
            mentions = data["mentions"]
            avg_sent = data["sentiment_sum"] / max(mentions, 1)
            score = mentions * 1.0 + max(0.0, avg_sent) * 2.0
            scored.append((sym, score, {**data, "sentiment": avg_sent}))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = []
        for sym, _score, data in scored[:max_results]:
            top.append({
                "symbol": data["symbol"],
                "name": data.get("name"),
                "mentions": data["mentions"],
                "sentiment": round(data["sentiment"], 3),
                "sources": sorted(list(data["sources"])),
                "excerpt": data.get("excerpt"),
            })
        return top


news_intelligence_service = NewsIntelligenceService()


