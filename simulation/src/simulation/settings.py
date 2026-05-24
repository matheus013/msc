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
}
