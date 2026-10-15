
# Mantem os caminhos locais ancorados no projeto ao executar este script.
import os as _os
from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    _os.chdir(PROJECT_ROOT)

import PyPDF2
import re
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description="Verifica citacoes e texto do PDF da qualificacao.")
parser.add_argument("pdf", nargs="?", type=_Path, default=PROJECT_ROOT.parent / "docs/qualification_presentation/main.pdf")
pdf_path = parser.parse_args().pdf


# Abrir e analisar o PDF
try:
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        print(f"Total de páginas: {num_pages}\n")

        # Variáveis para rastreamento
        broken_citations_pages = []
        pages_with_issues = defaultdict(list)
        all_text_content = []

        # Extrair texto de todas as páginas
        for page_num in range(num_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                all_text_content.append((page_num + 1, text))

                # Procurar por "?" isolado (citações quebradas)
                # Padrões de citação quebrada
                patterns = [
                    (r'\s\?\s', 'espaço ? espaço'),
                    (r'\s\?\.', '? seguido de ponto'),
                    (r',\s\?', '? seguido de vírgula'),
                    (r';\s\?', '? seguido de ponto-vírgula'),
                    (r'\[\s*\?\s*\]', '[?]'),
                    (r'\(\s*\?\s*\)', '(?)'),
                ]

                issue_found = False
                for pattern, description in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        issue_found = True
                        pages_with_issues[page_num + 1].append(f"{description}: {len(matches)}")

                if issue_found:
                    broken_citations_pages.append(page_num + 1)

                # Procurar por "cadeia ?" especificamente
                if 'cadeia ?' in text:
                    pages_with_issues[page_num + 1].append('CRÍTICO: "cadeia ?" encontrado')

            except Exception as e:
                print(f"Erro ao processar página {page_num + 1}: {e}")

        # Procurar por padrões no texto completo
        full_text = "\n".join([text for _, text in all_text_content])

        print(f"Páginas processadas: {len(all_text_content)}")
        print(f"\n--- RESULTADOS DE VALIDAÇÃO ---\n")

        # Verificar citações quebradas
        print("🔍 CITAÇÕES:")
        if broken_citations_pages:
            print(f"   ❌ Citações quebradas encontradas em {len(broken_citations_pages)} página(s):")
            for page_num in sorted(broken_citations_pages):
                print(f"      Página {page_num}: {', '.join(pages_with_issues[page_num])}")
        else:
            print("   ✅ Nenhuma citação quebrada encontrada")

        # Verificar "cadeia ?"
        if any('CRÍTICO' in v for v_list in pages_with_issues.values() for v in v_list):
            print("\n   ❌ CRÍTICO: 'cadeia ?' ainda presente no PDF")
        else:
            print("\n   ✅ 'cadeia ?' não encontrado (correção OK)")

        # Verificar caracteres especiais
        print("\n🔍 CARACTERES ESPECIAIS:")
        expected_chars = {
            '→': 'arrow (→)',
            '×': 'times (×)',
            'π': 'pi (π)',
            'μ': 'mu (μ)',
            '²': 'superscript 2',
            'β': 'beta (β)',
            'α': 'alpha (α)',
            'γ': 'gamma (γ)',
            'σ': 'sigma (σ)',
        }

        found_special_chars = {}
        for char, name in expected_chars.items():
            if char in full_text:
                count = full_text.count(char)
                found_special_chars[name] = count

        if found_special_chars:
            print("   ✅ Caracteres especiais encontrados:")
            for name, count in sorted(found_special_chars.items()):
                print(f"      {name}: {count} ocorrência(s)")
        else:
            print("   ⚠️  Nenhum caractere especial esperado encontrado")

        # Verificar caracteres acentuados português
        print("\n🔍 ACENTUAÇÃO PORTUGUESA:")
        accents = {'ã': 0, 'é': 0, 'ç': 0, 'ú': 0, 'ó': 0, 'á': 0, 'à': 0, 'õ': 0}
        for char in accents:
            accents[char] = full_text.count(char)

        accent_count = sum(accents.values())
        if accent_count > 0:
            print(f"   ✅ Acentos renderizados corretamente: {accent_count} total")
            for char, count in accents.items():
                if count > 0:
                    print(f"      '{char}': {count}")
        else:
            print("   ❌ PROBLEMA: Nenhum acento encontrado (possível issue de encoding)")

        # Resumo final
        print(f"\n--- RESUMO FINAL ---")
        print(f"Total de páginas: {num_pages}")
        print(f"Páginas com ? isolado: {len(broken_citations_pages)}")
        print(f"Status: {'❌ PROBLEMAS ENCONTRADOS' if broken_citations_pages else '✅ OK'}")

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {pdf_path}")
except Exception as e:
    print(f"Erro geral: {e}")
    import traceback
    traceback.print_exc()
