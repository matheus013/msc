"""
Kedro project settings — Kedro 1.x
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html
"""
HOOKS = ()

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "parameters": ["parameters*", "parameters/**", "**/parameters*"],
        "catalog":    ["catalog*",    "catalog/**",    "**/catalog*"],
        "logging":    ["logging*",    "logging/**",    "**/logging*"],
    },
    # merge_strategy padrao do Kedro e "destructive": quando um ambiente
    # (--env m5, --env bot, ...) redefine um bloco parcial de uma chave de
    # topo (ex.: so "data_ingestion.states"), a chave INTEIRA do base e
    # substituida, nao mesclada -- os ambientes m5/bot ficavam sem
    # cv_threshold, active_statuses, exclude_venda_tipos etc. do
    # conf/base/parameters/data_ingestion.yml (bug descoberto em
    # 2026-08-18: o benchmark M5 vinha rodando com cv_threshold=0,
    # desativando silenciosamente o filtro de intermitencia documentado em
    # REIMPLEMENTACAO_SOTA.md). "soft" faz merge recursivo por chave, como
    # os arquivos de ambiente sempre pressupuseram.
    "merge_strategy": {"parameters": "soft"},
}
