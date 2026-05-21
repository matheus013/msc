import os
import re
import requests
from bs4 import BeautifulSoup
import yaml
from llama_cpp import Llama
from typing import Optional, Dict, Any, List

BASE_URL = "https://menu.brendi.com.br/thesevenburger/"
OUTPUT_DIR = "output_yml"

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def text_to_ingredients(text: str) -> Optional[List[str]]:
    if not text:
        return None
    text = re.sub(r"A partir de R\$.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Serve.*", "", text, flags=re.IGNORECASE)
    if "," in text:
        parts = [p.strip() for p in re.split(r",| e | & ", text) if p.strip()]
        candidates = []
        for p in parts:
            if len(p) < 2:
                continue
            if re.search(r"R\$|\d+ml|\d+g", p):
                continue
            candidates.append(p)
        if candidates:
            return candidates
    return None

def parse_product_block(prod_soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    name_tag = prod_soup.find("h3")
    if not name_tag:
        return None
    name = name_tag.get_text().strip()
    descr = ""
    descr_tag = name_tag.find_next_sibling("p")
    if descr_tag:
        descr = descr_tag.get_text().strip()
    price = None
    text = prod_soup.get_text()
    m = re.search(r"R\$ *([\d.,]+)", text)
    if m:
        price = float(m.group(1).replace(".", "").replace(",", "."))
    ingredients = None
    ul = prod_soup.find("ul")
    if ul:
        items = [li.get_text().strip() for li in ul.find_all("li") if li.get_text().strip()]
        if items:
            ingredients = items
    if ingredients is None:
        # fallback: tentar extrair do texto da descrição
        ing_from_descr = text_to_ingredients(descr)
        if ing_from_descr:
            ingredients = ing_from_descr
    weight_g = None
    mm = re.search(r"(\d+)\s?g", name + " " + descr, re.IGNORECASE)
    if mm:
        try:
            weight_g = int(mm.group(1))
        except:
            weight_g = None
    low = (name + " " + descr).lower()
    contains_lactose = any(w in low for w in ["queijo", "prato", "cheese", "cream", "gorgonzola", "catupiry"])
    is_spicy = "pimenta" in low or "picante" in low or "spicy" in low
    is_vegan = "vegano" in low or "vegan" in low
    is_vegetarian = is_vegan or not any(w in low for w in ["meat", "burger", "bacon", "hambúrguer", "frango", "chicken"])
    return {
        "name": name,
        "description": descr,
        "price": price,
        "ingredients": ingredients,
        "weight_g": weight_g,
        "is_vegan": is_vegan,
        "is_vegetarian": is_vegetarian,
        "is_spicy": is_spicy,
        "contains_lactose": contains_lactose
    }

def load_llama(model_path: str) -> Llama:
    llm = Llama(model_path=model_path)
    return llm

def ask_llama_for_ingredients(llm: Llama, name: str, description: str) -> Dict[str, Any]:
    prompt = f"""
You are a food-product expert. Given the product name and a brief description/context, infer a plausible list of ingredients.

Name: "{name}"
Description: "{description}"

Return JSON with:
- ingredients: list of string or null
- weight_g: integer or null
- is_vegan: true/false/null
- is_vegetarian: true/false/null
- is_spicy: true/false/null
- contains_lactose: true/false/null

If uncertain, you may use null. Return only valid JSON.
"""
    # gerar resposta
    resp = llm(prompt, max_tokens=150, temperature=0.0)
    # dependendo da versão da biblioteca llama-cpp-python, acessar resp["choices"][0]["text"] ou resp["text"]
    text = None
    if "choices" in resp:
        text = resp["choices"][0]["text"].strip()
    else:
        text = resp["text"].strip()
    try:
        inferred = yaml.safe_load(text)
        if not isinstance(inferred, dict):
            raise ValueError("not dict")
    except Exception as e:
        print("Error parsing llama output:", e, text)
        inferred = {}
    return inferred

def merge_info(base: Dict[str, Any], inferred: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in inferred.items():
        if v is None:
            continue
        if k not in out or out[k] is None:
            out[k] = v
    return out

def generate_yaml(product_info: Dict[str, Any], path: str):
    if "id" not in product_info:
        product_info["id"] = slugify(product_info.get("name", "product"))
    product_info.setdefault("calories", None)
    product_info.setdefault("protein_g", None)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(product_info, f, allow_unicode=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # carregue o modelo llama local
    model_path = "path/to/your/model.gguf"  # ajuste para onde seu modelo está
    if not os.path.exists(model_path):
        print("LLama model not found at", model_path)
        print("Script vai continuar sem inferência por Llama.")
        llm = None
    else:
        llm = load_llama(model_path)
        print("Loaded llama model from", model_path)

    product_tags = soup.find_all("h3")
    for h3 in product_tags:
        container = h3
        for _ in range(3):
            if container is None:
                break
            if container.find("ul") or "R$" in container.get_text():
                break
            container = container.parent
        if container is None:
            container = h3.parent
        base = parse_product_block(container)
        if base is None:
            continue
        inferred = {}
        if llm is not None:
            inferred = ask_llama_for_ingredients(llm, base["name"], base.get("description", ""))
        merged = merge_info(base, inferred)
        fname = f"{slugify(merged['name'])}.yml"
        fullpath = os.path.join(OUTPUT_DIR, fname)
        generate_yaml(merged, fullpath)
        print("Wrote:", fullpath, merged)

    print("Done.")

if __name__ == "__main__":
    main()
