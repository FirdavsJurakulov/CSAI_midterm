"""Download openly-licensed car photos from Wikimedia Commons, labelled by brand (+ colour when known).

Brand comes from the category we crawl; colour is parsed from the file's own categories
(e.g. "White Jetour X70"), left empty when not stated. Author + licence are saved for attribution.

    python scripts/fetch_commons.py --out data/external/commons --per-brand 200
"""
import argparse, csv, json, os, random, re, time, urllib.parse, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

API = "https://commons.wikimedia.org/w/api.php"
UA = "CSAI3210-midterm-dataset/0.1 (student project; firdavs.j2006@gmail.com)"

# brand label -> Commons name used in "<name> automobiles"
BRANDS = {
    "BYD": "BYD", "Chery": "Chery", "Haval": "Haval", "Changan": "Changan", "Geely": "Geely",
    "Jetour": "Jetour", "Li Auto": "Li Auto", "Zeekr": "Zeekr", "Exeed": "Exeed", "Omoda": "Omoda",
    "Jaecoo": "Jaecoo", "Hongqi": "Hongqi", "Leapmotor": "Leapmotor", "JAC": "JAC", "Tank": "Tank",
    "Chevrolet": "Chevrolet", "Daewoo": "Daewoo", "Lada": "Lada", "Kia": "Kia", "Hyundai": "Hyundai",
    "Toyota": "Toyota", "Mercedes-Benz": "Mercedes-Benz", "BMW": "BMW", "Volkswagen": "Volkswagen",
    "Nissan": "Nissan", "Deepal": "Deepal", "Aion": "Aion", "Voyah": "Voyah",
    "AITO": "AITO", "Dongfeng": "Dongfeng",
}
# brands whose Commons tree does not follow "<name> automobiles"
EXTRA_SEEDS = {
    # current Chang'an line-up (the company category is dominated by 2000s Chana vans)
    "Changan": [f"Category:Chang'an {m}" for m in
                ["CS75 Plus", "CS75 Pro", "CS75", "CS55 Plus", "CS55 Plus (2nd generation)", "CS55",
                 "CS35 Plus", "UNI-K", "UNI-K iDD", "UNI-T", "UNI-V", "UNI-V iDD", "Eado DT",
                 "Eado (3rd generation)"]] + ["Category:Changan Hunter", "Category:Changan Lamore"],
    "Tank": ["Category:Tank 300", "Category:Tank 500", "Category:Tank 400", "Category:WEY Tank 300"],
    "Aion": ["Category:GAC Aion V", "Category:GAC Aion S", "Category:GAC Aion Y"],
}
# sub-brands / joint ventures inside a brand's tree that carry a different badge
EXCLUDE = {"Dongfeng": re.compile(r"voyah|nissan|honda|kia|renault|citro|infiniti|venucia", re.I),
           "Changan": re.compile(r"deepal|avatr", re.I)}
# brands whose old models (Nexia, Lacetti, Lada 2107, ...) still drive in Tashkent
OLD_MODELS_OK = {"Chevrolet", "Daewoo", "Lada", "Kia", "Hyundai", "Toyota", "Mercedes-Benz", "BMW",
                 "Volkswagen", "Nissan"}
# countries whose car fleet looks like Tashkent's are crawled first
REGIONS = ["Uzbekistan", "Kazakhstan", "Kyrgyzstan", "Russia", "Moscow", "Saint Petersburg", "China"]

SKIP = re.compile(r"interior|engine|logo|emblem|badge|concept|dashboard|seat|wheel|rim|detail|trunk|"
                  r"model car|scale model|toy|advert|factory|prototype|racing|race car|motorsport|"
                  r"crash|accident|wreck|police|taxi|bus|truck|van|steering|light|lamp|grille|"
                  r"\b(19\d\d|200[0-9])\b|by year|by decade|drawing|sketch|museum|"
                  r"mazda|suzuki|peugeot|citro|\bford\b|\bpsa\b", re.I)  # joint ventures carry other badges
COLOURS = [("white", "white"), ("black", "black"), ("grey", "grey"), ("gray", "grey"),
           ("silver", "grey"), ("red", "red"), ("maroon", "red"), ("blue", "blue"),
           ("green", "green"), ("brown", "brown"), ("beige", "brown"), ("golden", "brown"),
           ("gold", "brown"), ("bronze", "brown"), ("yellow", "yellow"), ("orange", "yellow")]


def api(**params):
    params.update(format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** attempt)
    return {}


def members(cat, kind):
    out, cont = [], {}
    while True:
        r = api(action="query", list="categorymembers", cmtitle=cat, cmtype=kind, cmlimit=500, **cont)
        out += [m["title"] for m in r.get("query", {}).get("categorymembers", [])]
        if "continue" not in r:
            return out
        cont = {"cmcontinue": r["continue"]["cmcontinue"]}


def crawl(name, limit, max_depth=4):
    """Breadth-first over the brand's category tree; regional categories first."""
    seeds = [f"Category:{name} automobiles in {c}" for c in REGIONS] + [f"Category:{name} automobiles"]
    seeds += EXTRA_SEEDS.get(name, []) + [f"Category:{name} vehicles"]
    seen, files = set(), []
    queue = deque((s, 0) for s in seeds)
    while queue and len(files) < limit * 3:
        cat, depth = queue.popleft()
        if cat in seen:
            continue
        seen.add(cat)
        files += [f for f in members(cat, "file") if f.lower().endswith((".jpg", ".jpeg"))]
        if depth < max_depth:
            subs = [s for s in members(cat, "subcat")
                    if not SKIP.search(s) and not (name in EXCLUDE and EXCLUDE[name].search(s))]
            random.shuffle(subs)
            queue.extend((s, depth + 1) for s in subs)
    files = list(dict.fromkeys(files))
    if name not in OLD_MODELS_OK:  # Chinese brands: models older than ~2016 are not on Tashkent roads
        files = [f for f in files if not re.search(r"\b(19\d\d|20(0\d|1[0-5]))\b", f)]
    random.shuffle(files)
    return files


def colour_of(categories):
    for c in categories:
        low = c.lower().removeprefix("category:")
        for word, label in COLOURS:
            if low.startswith(word + " "):
                return label
    return ""


def info(titles):
    r = api(action="query", prop="imageinfo|categories", titles="|".join(titles), iiprop="url|extmetadata",
            iiurlwidth=960, cllimit=500, clshow="!hidden")
    out = {}
    for p in r.get("query", {}).get("pages", []):
        ii = (p.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        out[p["title"]] = dict(
            url=ii.get("thumburl") or ii.get("url"),
            author=re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip(),
            license=meta.get("LicenseShortName", {}).get("value", ""),
            colour=colour_of([c["title"] for c in p.get("categories", [])]),
        )
    return out


def download(url, path):
    if os.path.exists(path):
        return True
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
            return True
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/commons")
    ap.add_argument("--per-brand", type=int, default=200)
    ap.add_argument("--brands", nargs="*", default=list(BRANDS))
    args = ap.parse_args()
    random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")
    new = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="", encoding="utf-8") as mf, ThreadPoolExecutor(4) as pool:
        w = csv.writer(mf)
        if new:
            w.writerow(["file", "brand", "colour", "title", "author", "license", "url"])
        for brand in args.brands:
            titles = crawl(BRANDS[brand], args.per_brand)[: args.per_brand]
            bdir = os.path.join(args.out, brand.replace(" ", "_"))
            os.makedirs(bdir, exist_ok=True)
            rows = []
            for i in range(0, len(titles), 50):
                for t, m in info(titles[i:i + 50]).items():
                    if m["url"]:
                        fname = re.sub(r"[^\w.-]", "_", t.removeprefix("File:"))[-120:]
                        rows.append((os.path.join(bdir, fname), t, m))
            ok = list(pool.map(lambda r: download(r[2]["url"], r[0]), rows))
            for (path, t, m), good in zip(rows, ok):
                if good:
                    w.writerow([os.path.relpath(path, args.out), brand, m["colour"], t, m["author"], m["license"], m["url"]])
            mf.flush()
            print(f"{brand:15s} {sum(ok):4d} images ({sum(1 for r in rows if r[2]['colour'])} with colour)", flush=True)


if __name__ == "__main__":
    main()
