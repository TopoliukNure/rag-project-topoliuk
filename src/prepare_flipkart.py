import argparse
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

from langdetect import DetectorFactory, LangDetectException, detect

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products"
DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/promptcloudhq/"
    "flipkart-products?datasetVersionNumber=1"
)
CSV_NAME = "flipkart_com-ecommerce_sample.csv"
CHECKSUM = "56f8f699c9e847356666c2eab3c3ab1244340f6a98ad08e39ea2199ebe993ad1"
SIZE, MIN_WORDS, WINDOW, OVERLAP = 1000, 200, 400, 40
PRICE_PHRASE = re.compile(
    r"(?:\bprice\s*:\s*)?(?:\b(?:rs\.?|inr|rupees)\s*|₹\s*)"
    r"(\d[\d,]*(?:\.\d+)?)(?!\w)|(\d[\d,]*(?:\.\d+)?)\s+(?:INR|rupees)\b",
    re.I,
)
RUBY_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|=>', re.S)
STOP_WORDS = set(
    "a an and are as at be by can for from has have in is it its of on or our "
    "that the their this to use was were which will with you your".split()
)


class PlainText(HTMLParser):
    blocks = {"p", "br", "div", "li", "tr", "td"}

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in self.blocks:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in self.blocks:
            self.parts.append(" ")


@dataclass
class Candidate:
    document: dict
    original_description: str
    removed_amounts: list[str]
    word_count: int
    rank: bytes


def clean(text):
    parser = PlainText()
    parser.feed(text)
    parser.close()
    normalized = unicodedata.normalize("NFC", "".join(parser.parts))
    return " ".join(normalized.split())


def remove_prices(text):
    return " ".join(PRICE_PHRASE.sub("", text).split())


def parse_specifications(raw):
    converted = RUBY_TOKEN.sub(
        lambda token: ":" if token[0] == "=>" else token[0], raw
    )
    items = json.loads(converted, strict=False)["product_specification"]
    if not isinstance(items, list):
        raise ValueError("Expected a list of specifications")
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Expected a specification object")
        key, value = item.get("key", ""), item.get("value", "")
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Expected string specification fields")
        key, value = clean(key), clean(value)
        pair = (key.casefold(), value.casefold())
        if pair in seen or pair[1] in {
            "", "na", "n/a", "not available", "none", "-"
        }:
            continue
        seen.add(pair)
        result.append({"key": key, "value": value})
    return result


def parse_price(raw):
    try:
        amount = Decimal(raw)
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    if amount != amount.to_integral_value():
        raise ValueError(f"Unexpected non-integer amount in the source: {raw}")
    return int(amount)


def read_source(local):
    if local is None:
        with urlopen(DOWNLOAD, timeout=60) as response:
            archive = io.BytesIO(response.read())
    else:
        archive = local
    if isinstance(archive, Path) and archive.suffix.lower() == ".csv":
        raw = archive.read_bytes()
    else:
        with ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("Invalid ZIP checksum")
            raw = bundle.read(CSV_NAME)
    if hashlib.sha256(raw).hexdigest() != CHECKSUM:
        raise ValueError(f"Unexpected SHA-256 for {CSV_NAME}")
    stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    reader = csv.DictReader(stream, strict=True)
    rows = list(reader)
    if len(rows) != 20_000 or len(reader.fieldnames or []) != 15:
        raise ValueError("Unexpected CSV dimensions")
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Unexpected CSV field count")
    return rows


def build_pools(rows):
    stats = Counter()
    pools = defaultdict(list)
    for source_row, row in enumerate(rows, 1):
        pid = row["pid"]
        if parse_qs(urlsplit(row["product_url"]).query).get("pid") != [pid]:
            raise ValueError(
                f"URL does not match PID at CSV record {source_row}"
            )
        title = clean(row["product_name"])
        description = clean(row["description"])
        raw_path = row["product_category_tree"]
        path = raw_path.removeprefix('["').removesuffix('"]')
        category = clean(path.split(" >> ", 1)[0])
        retail = parse_price(row["retail_price"])
        discounted = parse_price(row["discounted_price"])
        if retail is not None and discounted is not None:
            stats["priced"] += 1
        raw_specs = row["product_specifications"]
        if not raw_specs.strip():
            stats["missing_specifications"] += 1
            continue
        try:
            specs = parse_specifications(raw_specs)
        except (KeyError, TypeError, ValueError):
            stats["invalid_specifications"] += 1
            continue
        stats["parsed_specifications"] += 1
        if (
            retail is None
            or discounted is None
            or discounted > retail
            or not title
            or not description
            or not category
            or " >> " not in path
            or not any(spec["key"] for spec in specs)
        ):
            continue
        spec_lines = [
            f"{spec['key']}: {spec['value']}" if spec["key"] else spec["value"]
            for spec in specs
        ]
        parts = map(remove_prices, (title, description, *spec_lines))
        text = " ".join(filter(None, parts))
        if len(text) < 100:
            continue
        doc = {
            "document_id": f"flipkart:{pid}",
            "pid": pid,
            "uniq_id": row["uniq_id"],
            "source_row": source_row,
            "source_url": row["product_url"],
            "crawl_timestamp": row["crawl_timestamp"],
            "title": title,
            "category": category,
            "category_path": path,
            "language": "en",
            "currency": "INR",
            "retail_price": retail,
            "discounted_price": discounted,
            "specifications": specs,
            "text": text,
        }
        removed_amounts = [
            match[1] or match[2]
            for match in PRICE_PHRASE.finditer(description)
        ]
        pools[category].append(
            Candidate(
                document=doc,
                original_description=description,
                removed_amounts=removed_amounts,
                word_count=len(text.split()),
                rank=hashlib.sha256(pid.encode("utf-8")).digest(),
            )
        )
    stats["candidates"] = sum(map(len, pools.values()))
    return pools, stats


def normalized_key(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def select_documents(pools):
    DetectorFactory.seed = 0
    groups = deque()
    for _, items in sorted(pools.items()):
        if len(items) >= 10:
            eligible = (item for item in items if item.word_count >= MIN_WORDS)
            groups.append(deque(sorted(eligible, key=lambda item: item.rank)))
    products, titles, texts = set(), set(), set()
    selected = []
    while groups and len(selected) < SIZE:
        queue = groups.popleft()
        while queue:
            item = queue.popleft()
            doc = item.document
            pid = doc["pid"]
            title = normalized_key(doc["title"])
            text = normalized_key(doc["text"])
            if pid in products or title in titles or text in texts:
                continue
            try:
                language = detect(item.original_description)
            except LangDetectException:
                continue
            if language != "en":
                continue
            products.add(pid)
            titles.add(title)
            texts.add(text)
            selected.append(item)
            break
        if queue:
            groups.append(queue)
    categories = {item.document["category"] for item in selected}
    if len(selected) != SIZE or len(categories) != 28:
        raise ValueError(
            "The source did not reproduce the expected 1,000 products "
            "and 28 categories"
        )
    return selected


def chunks_for(document):
    words = document["text"].split()
    metadata = {
        key: value
        for key, value in document.items()
        if key not in {"text", "specifications"}
    }
    start, index = 0, 0
    while start < len(words):
        remaining = len(words) - start
        size = min(WINDOW, remaining)
        if WINDOW < remaining < WINDOW - OVERLAP + MIN_WORDS:
            size = remaining - MIN_WORDS + OVERLAP
        end = start + size
        yield {
            **metadata,
            "chunk_id": f"{document['document_id']}:{index:04d}",
            "chunk_index": index,
            "word_start": start,
            "word_end": end,
            "text": " ".join(words[start:end]),
        }
        if end == len(words):
            break
        start = end - OVERLAP
        index += 1


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")


def write_review(path, selected):
    sections = [
        "# Вибірка для ручної перевірки",
        "10 карток із фінального корпусу, seed = 2026. "
        "Номер запису CSV рахується від 1 після заголовка. У кожній картці "
        "наведено вихідні структуровані ціни, результат очищення та контроль "
        "вилучених цінових фраз.",
    ]
    for item in random.Random(2026).sample(selected, 10):
        doc = item.document
        amounts = [
            Decimal(amount.replace(",", "")) for amount in item.removed_amounts
        ]
        match_count = amounts.count(doc["discounted_price"])
        price_note = (
            ", ".join(item.removed_amounts)
            + f" INR; збігів зі зниженою ціною {match_count} "
              f"із {len(amounts)}."
            if amounts
            else "у полі description фрази цього формату не знайдено."
        )
        keyed_specs = [spec for spec in doc["specifications"] if spec["key"]]
        first_spec = keyed_specs[0]
        sections.extend([
            f"## {doc['pid']} — {doc['title']}",
            f"Категорія: {doc['category']}. Запис CSV: {doc['source_row']}. "
            f"[Оригінальний URL товару]({doc['source_url']}).",
            f"- Роздрібна ціна: {doc['retail_price']} INR. "
            f"Знижена: {doc['discounted_price']} INR.\n"
            f"- Слів після очищення: {item.word_count}. "
            f"Характеристик з ключем: {len(keyed_specs)}.\n"
            f"- Приклад характеристики: {first_spec['key']} — "
            f"{first_spec['value']}.\n"
            f"- Вилучені цінові фрази з опису: {price_note}",
            f"~~~~text\n{doc['text']}\n~~~~",
        ])
    review = "\n\n".join(sections) + "\n"
    path.write_text(review, encoding="utf-8", newline="\n")


def write_data_card(path, rows, selected, chunks, stats):
    docs = [item.document for item in selected]
    sizes = [item.word_count for item in selected]
    lengths = [item["word_end"] - item["word_start"] for item in chunks]
    categories = Counter(doc["category"] for doc in docs)
    words = Counter(
        word
        for doc in docs
        for word in re.findall(r"[a-z]+(?:'[a-z]+)?", doc["text"].lower())
        if word not in STOP_WORDS
    )
    top = sorted(words.items(), key=lambda pair: (-pair[1], pair[0]))[:20]
    top_rows = "\n".join(f"| {word} | {count} |" for word, count in top)
    category_counts = ", ".join(
        f"{key} ({count})" for key, count in categories.most_common(8)
    )
    discounts = [doc["discounted_price"] for doc in docs]
    retail = [doc["retail_price"] for doc in docs]
    with_amount = [item for item in selected if item.removed_amounts]
    conflicts = sum(
        any(
            Decimal(amount.replace(",", ""))
            != item.document["discounted_price"]
            for amount in item.removed_amounts
        )
        for item in with_amount
    )
    text = f"""# Картка даних Smart Product Finder — Flipkart Products

## Джерело й ліцензія

[Flipkart Products, PromptCloudHQ, версія 1]({SOURCE}).
Автор набору — PromptCloud, товарні картки зібрано з Flipkart.com.
Ліцензія даних — [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Файл: {CSV_NAME}. SHA-256: {CHECKSUM}.

Період збору за crawl_timestamp: {min(row['crawl_timestamp'] for row in rows)}
— {max(row['crawl_timestamp'] for row in rows)}. Ціни історичні. Валюту INR
визначено за позначенням Rs. у вихідних описах і документацією Flipkart.
CSV не містить окремого поля валюти.
[Документація Flipkart](https://affiliate.flipkart.com/api-docs/af_prod_ref.html).

ABO був джерелом у проєкті лабораторної № 1. Із лабораторної № 2 основним
корпусом для індексування, пошуку і тестування є Flipkart Products.

## Підготовка та статистика

Нормалізовано Unicode до NFC і пробіли, вилучено HTML і грошові суми з позначеннями Rs., INR, rupees та ₹ з пошукового тексту. Пари ключ–значення декодовано з формату
Ruby hash, повтори й порожні значення прибрано. Пошуковий документ об'єднує
назву, опис і характеристики. Структуровані ціни перенесено з CSV в INR
без конвертації. Для фільтра бюджету використовують поле discounted_price.

Кандидати мають англомовний опис за langdetect, шлях категорії, характеристики,
обидві додатні ціни, щонайменше 100 символів і 200 слів. Відбір у категоріях
з від 10 кандидатів відбувається циклічно. Усередині категорії порядок
визначає SHA-256 від pid. Дублікати за pid, нормалізованою назвою або
пошуковим текстом відсіюються.

| Показник | Значення |
| --- | ---: |
| Рядків вихідного CSV | {len(rows)} |
| Унікальних вихідних pid | {len({row['pid'] for row in rows})} |
| Рядків із двома додатними цінами | {stats['priced']} |
| Розібраних / порожніх / некоректних специфікацій | {stats['parsed_specifications']} / {stats['missing_specifications']} / {stats['invalid_specifications']} |
| Кандидатів до перевірки мови, дедуплікації та порога 200 слів | {stats['candidates']} |
| Документів / верхніх категорій | {len(docs)} / {len(categories)} |
| Довжина документа, мін. / макс. / середня (слів) | {min(sizes)} / {max(sizes)} / {mean(sizes):.2f} |
| Чанків | {len(chunks)} |
| Довжина чанка, мін. / макс. / середня (слів) | {min(lengths)} / {max(lengths)} / {mean(lengths):.2f} |
| Чанків коротших за 200 слів | {sum(size < MIN_WORDS for size in lengths)} |
| Роздрібні ціни, мін. / макс. (INR) | {min(retail)} / {max(retail)} |
| Знижені ціни, мін. / макс. (INR) | {min(discounts)} / {max(discounts)} |
| Карток із вилученими ціновими фразами в описі | {len(with_amount)} |
| Із них карток із розбіжністю з discounted_price | {conflicts} |

Найчисельніші категорії: {category_counts}.
Вікно має до {WINDOW} слів. Сусідні чанки перекриваються на {OVERLAP} слів.
Для короткого залишку скорочується передостаннє вікно. Стабільний document_id
утворюється з pid, chunk_id додає порядковий номер із нуля. word_start
включений, word_end виключений. Усі характеристики збережено в documents.jsonl.

## Топ-20 слів

Частоти пораховано за очищеними документами до чанкування. Регістр зведено
до нижнього, поширені службові слова вилучено.

| Слово | Кількість |
| --- | ---: |
{top_rows}

## Обмеження

- Ціни й характеристики відображають історичний знімок, наявність товарів
  і чинність URL не перевірялися.
- Відбір надає перевагу довгим карткам. Розподіл категорій відрізняється від
  вихідного CSV. Споріднені варіанти можуть мати різні pid.
- Мовний детектор працює на описі. Окремі назви та характеристики можуть
  містити власні назви, фрагменти іншими мовами, маркетингові повтори або
  суперечності. Непідтверджені характеристики не виправлялися.
- Десять карток і результати очищення наведено в [manual-review.md](manual-review.md).

Відтворення з кореня репозиторію: python src/prepare_flipkart.py.
Для перевіреного локального ZIP або CSV використайте параметр --source PATH.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the version 1 Flipkart Products corpus for Lab 2."
    )
    parser.add_argument(
        "--source", type=Path, help="Original version 1 Kaggle ZIP or CSV"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = read_source(args.source)
    pools, stats = build_pools(rows)
    selected = select_documents(pools)
    documents = [item.document for item in selected]
    chunks = [chunk for doc in documents for chunk in chunks_for(doc)]
    data = args.output_root / "data/lab-2"
    reports = args.output_root / "reports/lab-2"
    data.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_jsonl(data / "documents.jsonl", documents)
    write_jsonl(data / "chunks.jsonl", chunks)
    write_review(reports / "manual-review.md", selected)
    write_data_card(reports / "data-card.md", rows, selected, chunks, stats)
    print(f"Prepared {len(documents)} documents and {len(chunks)} chunks.")


if __name__ == "__main__":
    main()

