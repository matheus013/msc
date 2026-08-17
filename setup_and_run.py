import subprocess
import sys

# Instalar dependências
packages = ["pandas", "numpy", "pyyaml", "scikit-learn", "xgboost"]
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f"✓ {pkg} já está instalado")
    except:
        print(f"📦 Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("\n✅ Todas as dependências instaladas\n")

# Agora executar transform_data.py
exec(open("transform_data.py").read())
